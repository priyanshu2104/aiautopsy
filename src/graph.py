"""
LangGraph orchestrator for AI Autopsy.

Defines:
  - AutopsyState: shared state flowing through all 3 agents
  - investigator_node: wraps InvestigatorPipeline as an agent node
  - build_graph(): compiles and returns the runnable LangGraph app

Usage:
    from src.graph import build_graph
    app = build_graph()
    result = app.invoke({
        "model_path": "models/fraud_rf.pkl",
        "csv_path": "data/mispredictions/fraud_wrong.csv",
        "model_name": "Fraud Detector",
        "investigator_output": None,
        "counterfactual_output": None,
        "report_output": None,
        "error": None
    })
    print(result["investigator_output"])
"""
import logging
import time
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END

logger = logging.getLogger(__name__)


# ── Shared state ──────────────────────────────────────────────────────────────

class AutopsyState(TypedDict):
    """
    Flows through every agent node.
    Each agent reads from it and writes its output back into it.
    Never mutate state directly — always return {**state, key: value}
    """
    model_path: str
    csv_path: str
    model_name: str
    investigator_output: Optional[dict]
    counterfactual_output: Optional[dict]
    report_output: Optional[dict]
    error: Optional[str]
    timing: Optional[dict]


# ── Agent nodes ───────────────────────────────────────────────────────────────

def investigator_node(state: AutopsyState) -> AutopsyState:
    """Agent 1: Runs SHAP analysis on mispredictions."""
    logger.info(f"Agent 1 starting — model: {state['model_name']}")
    t0 = time.time()
    try:
        from src.investigator import InvestigatorPipeline
        result = InvestigatorPipeline().run(
            model_path=state["model_path"],
            mispredictions_path=state["csv_path"],
            model_name=state["model_name"],
        )
        elapsed = round(time.time() - t0, 2)
        logger.info(
            f"Agent 1 complete — {result.get('total_failures', 0)} failures found"
            f"failures in {elapsed}s")
        timing = state.get("timing") or {}
        timing["agent1_s"] = elapsed
        return {**state, "investigator_output": result, "timing": timing}
    
    except Exception as e:
        elapsed = round(time.time() - t0, 2)
        logger.error(f"[Agent 1] Failed after {elapsed}s: {e}")
        return {**state,
                "error": f"Agent 1 (investigator) failed: {str(e)}"}


def counterfactual_node(state: AutopsyState) -> AutopsyState:
    """
    Agent 2: Generates counterfactuals for top mispredictions.
    Uses InvestigatorPipeline output (from Agent 1) to prioritise
    which features to perturb.
    """
    logger.info("Agent 2 starting — counterfactual generation")
    t0 = time.time()
    try:
        from src.counterfactual import CounterfactualPipeline
        result = CounterfactualPipeline().run(
            model_path=state["model_path"],
            mispredictions_path=state["csv_path"],
            investigator_output=state.get("investigator_output", {}),
            top_n=10,
        )
        elapsed = round(time.time() - t0, 2)
        logger.info(
            f"Agent 2 complete — "
            f"{result.get('found', 0)}/{result.get('attempted', 0)} "
            f"CFs in {elapsed}s")
        timing = state.get("timing") or {}
        timing["agent2_s"] = elapsed
        return {**state, "counterfactual_output": result, "timing": timing}

    except Exception as e:
        elapsed = round(time.time() - t0, 2)
        logger.error(f"[Agent 2] Failed after {elapsed}s: {e}")
        # Don't fail the whole pipeline if CF fails
        # Return empty result so Agent 3 can still run
        return {
            **state,
            "counterfactual_output": {
                "agent": "counterfactual",
                "error": str(e),
                "attempted": 0,
                "found": 0,
                "success_rate": 0.0,
                "avg_features_to_flip": 0.0,
                "examples": [],
            }
        }


def reporter_node(state: AutopsyState) -> AutopsyState:
    """Agent 3: LLM report generation (Week 5 — placeholder for now)."""
    logger.info("Agent 3 — reporter (placeholder for Week 5)")
    return {
        **state,
        "report_output": {
            "agent": "reporter",
            "version": "placeholder",
            "note": "Agent 3 not implemented yet — coming in Week 5"
        }
    }


# ── Routing ───────────────────────────────────────────────────────────────────

def should_run_counterfactual(state: AutopsyState) -> str:
    """
    After Agent 1 runs, decide what to do next.
    Returns 'counterfactual' to continue or 'end' to stop.
    """
    if state.get("error"):
        logger.warning(f"Stopping due to error: {state['error']}")
        return "end"

    inv = state.get("investigator_output", {})
    if not inv or inv.get("total_failures", 0) == 0:
        logger.info("No failures found — skipping remaining agents")
        return "end"

    return "counterfactual"

def should_run_reporter(state: AutopsyState) -> str:
    """After Agent 2: always run reporter (even if CF found nothing)."""
    # CF errors are non-fatal — reporter still runs
    return "reporter"


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_graph():
    """
    Builds and compiles the LangGraph state machine.
    Returns a compiled app ready for app.invoke({...})
    """
    graph = StateGraph(AutopsyState)

    graph.add_node("investigator",   investigator_node)
    graph.add_node("counterfactual", counterfactual_node)
    graph.add_node("reporter",       reporter_node)

    graph.set_entry_point("investigator")

    graph.add_conditional_edges(
        "investigator",
        should_run_counterfactual,
        {"counterfactual": "counterfactual", "end": END}
    )
    graph.add_conditional_edges(
        "counterfactual",
        should_run_reporter,
        {"reporter": "reporter"}
    )
    graph.add_edge("counterfactual", "reporter")
    graph.add_edge("reporter", END)

    return graph.compile()


def make_initial_state(
    model_path: str,
    csv_path: str,
    model_name: str = "Unknown Model"
) -> AutopsyState:
    """Helper to create a clean initial state dict."""
    return {
        "model_path": model_path,
        "csv_path": csv_path,
        "model_name": model_name,
        "investigator_output": None,
        "counterfactual_output": None,
        "report_output": None,
        "error": None,
        "timing": {},
    }