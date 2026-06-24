"""
Agent 1: Data Investigator

Takes any sklearn model + a CSV of its mispredictions.
Returns a structured JSON dict identifying which features drive failures
and how they cluster.

Usage (standalone):
    from src.investigator import InvestigatorPipeline
    pipeline = InvestigatorPipeline()
    result = pipeline.run("models/fraud_rf.pkl",
                          "data/mispredictions/fraud_wrong.csv",
                          model_name="Fraud Detector")
    print(result)
"""
import shap
import pandas as pd
import numpy as np
import joblib
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class InvestigatorPipeline:
    """
    Wraps SHAP analysis into a clean interface for the LangGraph agent.
    Designed to work with any binary sklearn classifier that has
    a predict() method.
    """

    def run(
        self,
        model_path: str,
        mispredictions_path: str,
        model_name: str = "unknown",
        top_n_features: int = 3,
        top_n_patterns: int = 5,
    ) -> dict:
        """
        Run the full investigation on a model's mispredictions.

        Args:
            model_path: path to a .pkl joblib-saved sklearn model
            mispredictions_path: path to CSV with columns:
                                 [features..., predicted, actual]
            model_name: human-readable name for the report
            top_n_features: how many top SHAP features to return
            top_n_patterns: how many failure range patterns to return

        Returns:
            dict matching the InvestigatorOutput schema in schema.md
        """
        try:
            model = self._load_model(model_path)
            wrong, feature_cols = self._load_mispredictions(
                mispredictions_path)

            if len(wrong) == 0:
                return self._empty_result(model_name)

            shap_values = self._compute_shap(model, wrong[feature_cols])
            top_features = self._top_features(
                shap_values, feature_cols, top_n_features)
            patterns = self._failure_patterns(
                wrong, feature_cols, top_features, top_n_patterns)

            return {
                "agent": "investigator",
                "version": "1.0",
                "model": model_name,
                "total_failures": len(wrong),
                "top_features": top_features,
                "failure_patterns": patterns,
            }

        except FileNotFoundError as e:
            logger.error(f"File not found: {e}")
            return {"agent": "investigator", "error": str(e),
                    "total_failures": 0, "top_features": [],
                    "failure_patterns": []}
        except Exception as e:
            logger.error(f"InvestigatorPipeline failed: {e}")
            return {"agent": "investigator", "error": str(e),
                    "total_failures": 0, "top_features": [],
                    "failure_patterns": []}

    # ── Private helpers ───────────────────────────────────────────────────────

    def _load_model(self, model_path: str):
        """Load a joblib-saved sklearn model."""
        model = joblib.load(model_path)
        if not hasattr(model, "predict"):
            raise ValueError(
                f"Model at {model_path} has no predict() method. "
                "Only sklearn-compatible models are supported.")
        return model

    def _load_mispredictions(self, path: str):
        """
        Load mispredictions CSV. Expects columns: [features..., predicted, actual]
        Returns (dataframe, list_of_feature_column_names)
        """
        df = pd.read_csv(path)
        if "predicted" not in df.columns or "actual" not in df.columns:
            raise ValueError(
                f"CSV at {path} must have 'predicted' and 'actual' columns. "
                f"Found columns: {list(df.columns)}")

        feature_cols = [c for c in df.columns
                        if c not in ("predicted", "actual")]
        # Keep only numeric feature columns (SHAP needs floats)
        numeric_cols = df[feature_cols].select_dtypes(
            include=[np.number]).columns.tolist()
        return df, numeric_cols

    def _compute_shap(self, model, X: pd.DataFrame) -> np.ndarray:
        """
        Compute SHAP values for all rows.
        Returns a 2D numpy array: shape (n_rows, n_features)
        """
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        # For binary classifiers, shap_values is a list of 2 arrays
        # We want class 1 (positive/failure class)
        if isinstance(shap_values, list):
            return shap_values[1]
        return shap_values

    def _top_features(
        self,
        shap_values: np.ndarray,
        feature_cols: list,
        top_n: int,
    ) -> list:
        """
        Returns the top N features by mean absolute SHAP across all rows.
        These are the features that globally drive the most failures.
        """
        mean_abs = np.abs(shap_values).mean(axis=0)
        top_idx = np.argsort(mean_abs)[::-1][:top_n]
        return [
            {
                "feature": str(feature_cols[i]),
                "mean_abs_shap": round(float(mean_abs[i]), 4),
            }
            for i in top_idx
        ]

    def _failure_patterns(
        self,
        wrong: pd.DataFrame,
        feature_cols: list,
        top_features: list,
        top_n: int,
    ) -> list:
        """
        For each top feature, finds which value ranges have the most failures.
        e.g. "MonthlyCharges in (70, 90] has 45% failure rate"
        """
        patterns = []
        for feat_info in top_features:
            feat = feat_info["feature"]
            if feat not in wrong.columns:
                continue
            try:
                bins = pd.cut(wrong[feat], bins=5)
                counts = wrong.groupby(bins, observed=True).size()
                total = len(wrong)
                for b, count in counts.sort_values(ascending=False).items():
                    if count > 2:
                        patterns.append({
                            "feature": feat,
                            "range": str(b),
                            "failure_count": int(count),
                            "failure_rate": round(float(count) / total, 3),
                        })
            except Exception as e:
                logger.warning(f"Could not bin feature {feat}: {e}")

        return patterns[:top_n]

    def _empty_result(self, model_name: str) -> dict:
        """Returned when the mispredictions CSV has zero rows."""
        return {
            "agent": "investigator",
            "version": "1.0",
            "model": model_name,
            "total_failures": 0,
            "top_features": [],
            "failure_patterns": [],
            "note": "No mispredictions found — model may be performing perfectly on this test set.",
        }
        