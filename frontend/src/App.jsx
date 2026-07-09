import { useState, useRef } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip,
  ResponsiveContainer, Cell
} from 'recharts'

const API_URL = 'http://localhost:8000'
const COLORS = ['#1D9E75', '#5DCAA5', '#9FE1CB']

// ── Sub-components ────────────────────────────────────────────────────────────

function AgentProgress({ status, investigator, counterfactual, report }) {
  const agents = [
    {
      name: 'Agent 1 — Investigator',
      done: !!investigator,
      detail: investigator
        ? `${investigator.total_failures} failures found`
        : 'SHAP analysis...',
    },
    {
      name: 'Agent 2 — Counterfactual',
      done: !!counterfactual,
      detail: counterfactual ? 'Complete' : 'Waiting...',
    },
    {
      name: 'Agent 3 — Reporter',
      done: !!report,
      detail: report ? 'Complete' : 'Waiting...',
    },
  ]

  if (status === 'idle') return null

  return (
    <div style={styles.progressCard}>
      <h3 style={styles.cardTitle}>Pipeline progress</h3>
      {agents.map((agent, i) => (
        <div key={i} style={styles.agentRow}>
          <span style={{
            ...styles.agentDot,
            background: agent.done ? '#1D9E75' : '#D3D1C7'
          }} />
          <span style={styles.agentName}>{agent.name}</span>
          <span style={styles.agentDetail}>{agent.detail}</span>
        </div>
      ))}
    </div>
  )
}

function InvestigatorResults({ data }) {
  if (!data) return null

  const chartData = data.top_features.map(f => ({
    name: f.feature,
    shap: parseFloat(f.mean_abs_shap.toFixed(4)),
  }))

  return (
    <div style={styles.resultsCard}>
      <h3 style={styles.cardTitle}>
        Agent 1 results — {data.model}
      </h3>

      <div style={styles.statRow}>
        <div style={styles.stat}>
          <div style={styles.statNum}>{data.total_failures}</div>
          <div style={styles.statLabel}>Total failures</div>
        </div>
        <div style={styles.stat}>
          <div style={styles.statNum}>{data.top_features.length}</div>
          <div style={styles.statLabel}>Top features</div>
        </div>
        <div style={styles.stat}>
          <div style={styles.statNum}>{data.failure_patterns.length}</div>
          <div style={styles.statLabel}>Failure patterns</div>
        </div>
      </div>

      <h4 style={styles.subTitle}>Top failure-driving features (SHAP)</h4>
      <ResponsiveContainer width="100%" height={180}>
        <BarChart data={chartData} layout="vertical"
          margin={{ left: 20, right: 20 }}>
          <XAxis type="number" tick={{ fontSize: 11 }} />
          <YAxis dataKey="name" type="category"
            tick={{ fontSize: 11 }} width={80} />
          <Tooltip formatter={(v) => v.toFixed(4)} />
          <Bar dataKey="shap" radius={[0, 4, 4, 0]}>
            {chartData.map((_, i) => (
              <Cell key={i} fill={COLORS[i % COLORS.length]} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>

      <h4 style={styles.subTitle}>Failure patterns</h4>
      <table style={styles.table}>
        <thead>
          <tr>
            {['Feature', 'Range', 'Count', 'Rate'].map(h => (
              <th key={h} style={styles.th}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.failure_patterns.map((p, i) => (
            <tr key={i} style={i % 2 === 0 ? styles.trEven : {}}>
              <td style={styles.td}>{p.feature}</td>
              <td style={styles.td}><code>{p.range}</code></td>
              <td style={styles.td}>{p.failure_count}</td>
              <td style={styles.td}>
                <span style={{
                  ...styles.rateBadge,
                  background: p.failure_rate > 0.3 ? '#FFEBEB' : '#E1F5EE',
                  color: p.failure_rate > 0.3 ? '#CC2222' : '#085041',
                }}>
                  {(p.failure_rate * 100).toFixed(1)}%
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function CounterfactualResults({ data }) {
  if (!data || data.found === 0) return null

  // Filter out placeholder from week 2
  if (data.version === 'placeholder') return null

  return (
    <div style={styles.resultsCard}>
      <h3 style={styles.cardTitle}>
        Agent 2 results — counterfactual analysis
      </h3>

      <div style={styles.statRow}>
        <div style={styles.stat}>
          <div style={styles.statNum}>{data.found}</div>
          <div style={styles.statLabel}>Found</div>
        </div>
        <div style={styles.stat}>
          <div style={styles.statNum}>
            {(data.success_rate * 100).toFixed(0)}%
          </div>
          <div style={styles.statLabel}>Success rate</div>
        </div>
        <div style={styles.stat}>
          <div style={styles.statNum}>{data.avg_features_to_flip}</div>
          <div style={styles.statLabel}>Avg features to flip</div>
        </div>
      </div>

      <h4 style={styles.subTitle}>What would have fixed each prediction?</h4>
      {data.examples.slice(0, 5).map((ex, i) => {
        const feat = ex.features_changed[0]
        const delta = ex.delta[feat]
        const orig = ex.original_value[feat]
        const cfVal = ex.counterfactual_value[feat]
        const dir = delta > 0 ? 'increased' : 'decreased'

        return (
          <div key={i} style={styles.cfBox}>
            <span style={styles.cfFeat}>{feat}</span>
            {' '}{dir} by{' '}
            <strong>{Math.abs(delta).toFixed(2)}</strong>
            {' '}({ex.pct_change}% change,{' '}
            {orig.toFixed(2)} → {cfVal.toFixed(2)})
          </div>
        )
      })}

      <p style={styles.cfNote}>
        {data.avg_features_to_flip <= 1.5
          ? '⚠️ Model is brittle in this region — small changes flip predictions'
          : '✓ Model is moderately robust — larger changes needed to flip predictions'}
      </p>
    </div>
  )
}

// ── Main App ──────────────────────────────────────────────────────────────────

export default function App() {
  const [modelFile, setModelFile] = useState(null)
  const [csvFile, setCsvFile] = useState(null)
  const [modelName, setModelName] = useState('')
  const [status, setStatus] = useState('idle')
  // status: idle | uploading | running | complete | failed
  const [jobId, setJobId] = useState(null)
  const [results, setResults] = useState(null)
  const [errorMsg, setErrorMsg] = useState('')
  const pollRef = useRef(null)

  const isReady = modelFile && csvFile

  function stopPolling() {
    if (pollRef.current) {
      clearInterval(pollRef.current)
      pollRef.current = null
    }
  }

  async function pollResults(id) {
    try {
      const res = await axios.get(`${API_URL}/results/${id}`)
      const data = res.data
      setResults(data)

      if (data.status === 'complete') {
        setStatus('complete')
        stopPolling()
      } else if (data.status === 'failed') {
        setStatus('failed')
        setErrorMsg(data.error || 'Analysis failed')
        stopPolling()
      }
    } catch (e) {
      console.error('Poll error:', e)
    }
  }

  async function handleSubmit() {
    if (!isReady) return
    setStatus('uploading')
    setErrorMsg('')
    setResults(null)
    stopPolling()

    try {
      const form = new FormData()
      form.append('model_file', modelFile)
      form.append('csv_file', csvFile)
      form.append('model_name',
        modelName || modelFile.name.replace('.pkl', ''))

      const res = await axios.post(`${API_URL}/analyze`, form)
      const { job_id } = res.data
      setJobId(job_id)
      setStatus('running')

      // Poll every 3 seconds
      pollRef.current = setInterval(() => pollResults(job_id), 3000)
      // Also poll immediately
      pollResults(job_id)

    } catch (e) {
      setStatus('failed')
      setErrorMsg(
        e.response?.data?.detail || 'Upload failed. Is the API running?')
    }
  }

  function handleReset() {
    stopPolling()
    setModelFile(null)
    setCsvFile(null)
    setModelName('')
    setStatus('idle')
    setJobId(null)
    setResults(null)
    setErrorMsg('')
  }

  // ── Render ──────────────────────────────────────────────────────────────────
  return (
    <div style={styles.page}>
      <div style={styles.container}>

        {/* Header */}
        <div style={styles.header}>
          <h1 style={styles.title}>AI Autopsy</h1>
          <p style={styles.subtitle}>
            Autonomous ML failure investigation
          </p>
        </div>

        {/* Upload card */}
        <div style={styles.uploadCard}>
          <h3 style={styles.cardTitle}>Upload your model</h3>

          <label style={styles.label}>Model file (.pkl)</label>
          <input
            type="file"
            accept=".pkl"
            style={styles.fileInput}
            onChange={e => setModelFile(e.target.files[0])}
            disabled={status === 'running' || status === 'uploading'}
          />
          {modelFile && (
            <p style={styles.fileName}>✓ {modelFile.name}</p>
          )}

          <label style={styles.label}>Mispredictions CSV</label>
          <input
            type="file"
            accept=".csv"
            style={styles.fileInput}
            onChange={e => setCsvFile(e.target.files[0])}
            disabled={status === 'running' || status === 'uploading'}
          />
          {csvFile && (
            <p style={styles.fileName}>✓ {csvFile.name}</p>
          )}

          <label style={styles.label}>Model name (optional)</label>
          <input
            type="text"
            placeholder="e.g. Credit Card Fraud Detector"
            style={styles.textInput}
            value={modelName}
            onChange={e => setModelName(e.target.value)}
            disabled={status === 'running' || status === 'uploading'}
          />

          <div style={styles.buttonRow}>
            <button
              style={{
                ...styles.button,
                opacity: (!isReady || status === 'running' ||
                  status === 'uploading') ? 0.5 : 1,
                cursor: (!isReady || status === 'running' ||
                  status === 'uploading') ? 'not-allowed' : 'pointer',
              }}
              onClick={handleSubmit}
              disabled={!isReady || status === 'running' ||
                status === 'uploading'}
            >
              {status === 'uploading' ? 'Uploading...'
                : status === 'running' ? 'Analysing...'
                  : 'Run Autopsy'}
            </button>

            {(status === 'complete' || status === 'failed') && (
              <button style={styles.resetButton} onClick={handleReset}>
                New analysis
              </button>
            )}
          </div>

          {/* Status messages */}
          {status === 'running' && (
            <p style={styles.statusMsg}>
              ⏳ Analysis running... (auto-refreshing every 3s)
            </p>
          )}
          {status === 'failed' && (
            <p style={{ ...styles.statusMsg, color: '#CC2222' }}>
              ❌ {errorMsg}
            </p>
          )}
        </div>

        {/* Agent progress */}
        <AgentProgress
          status={status}
          investigator={results?.investigator}
          counterfactual={results?.counterfactual}
          report={results?.report}
        />

        {/* Results */}
        {results?.investigator && (
          <>
            <InvestigatorResults data={results.investigator} />
            <CounterfactualResults data={results?.counterfactual} />
          </>
        )}

      </div>
    </div>
  )
}

// ── Styles ────────────────────────────────────────────────────────────────────

const styles = {
  page: {
    minHeight: '100vh',
    background: '#F8F9FA',
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
    padding: '24px 16px',
  },
  container: {
    maxWidth: 720,
    margin: '0 auto',
  },
  header: {
    marginBottom: 24,
  },
  title: {
    fontSize: 28,
    fontWeight: 600,
    color: '#111',
    margin: '0 0 4px',
  },
  subtitle: {
    fontSize: 14,
    color: '#666',
    margin: 0,
  },
  uploadCard: {
    background: '#fff',
    border: '1px solid #E5E7EB',
    borderRadius: 12,
    padding: '20px 24px',
    marginBottom: 16,
  },
  progressCard: {
    background: '#fff',
    border: '1px solid #E5E7EB',
    borderRadius: 12,
    padding: '16px 24px',
    marginBottom: 16,
  },
  resultsCard: {
    background: '#fff',
    border: '1px solid #E5E7EB',
    borderRadius: 12,
    padding: '20px 24px',
    marginBottom: 16,
  },
  cardTitle: {
    fontSize: 15,
    fontWeight: 600,
    color: '#111',
    margin: '0 0 16px',
  },
  label: {
    display: 'block',
    fontSize: 13,
    fontWeight: 500,
    color: '#374151',
    marginBottom: 6,
    marginTop: 12,
  },
  fileInput: {
    display: 'block',
    width: '100%',
    fontSize: 13,
    color: '#374151',
    cursor: 'pointer',
  },
  textInput: {
    display: 'block',
    width: '100%',
    fontSize: 13,
    padding: '8px 10px',
    border: '1px solid #D1D5DB',
    borderRadius: 6,
    outline: 'none',
    boxSizing: 'border-box',
    color: '#111',
    marginTop: 0,
  },
  fileName: {
    fontSize: 12,
    color: '#1D9E75',
    margin: '4px 0 0',
  },
  buttonRow: {
    display: 'flex',
    gap: 10,
    marginTop: 20,
  },
  button: {
    background: '#1D9E75',
    color: '#fff',
    border: 'none',
    borderRadius: 8,
    padding: '10px 24px',
    fontSize: 14,
    fontWeight: 500,
  },
  resetButton: {
    background: 'none',
    color: '#6B7280',
    border: '1px solid #D1D5DB',
    borderRadius: 8,
    padding: '10px 20px',
    fontSize: 14,
    cursor: 'pointer',
  },
  statusMsg: {
    fontSize: 13,
    color: '#6B7280',
    marginTop: 12,
  },
  agentRow: {
    display: 'flex',
    alignItems: 'center',
    gap: 10,
    padding: '6px 0',
    borderBottom: '1px solid #F3F4F6',
  },
  agentDot: {
    width: 8,
    height: 8,
    borderRadius: '50%',
    flexShrink: 0,
  },
  agentName: {
    fontSize: 13,
    color: '#374151',
    flex: 1,
  },
  agentDetail: {
    fontSize: 12,
    color: '#9CA3AF',
  },
  statRow: {
    display: 'grid',
    gridTemplateColumns: 'repeat(3, 1fr)',
    gap: 12,
    marginBottom: 20,
  },
  stat: {
    background: '#F9FAFB',
    borderRadius: 8,
    padding: '12px',
    textAlign: 'center',
  },
  statNum: {
    fontSize: 22,
    fontWeight: 600,
    color: '#111',
  },
  statLabel: {
    fontSize: 11,
    color: '#6B7280',
    marginTop: 2,
  },
  subTitle: {
    fontSize: 13,
    fontWeight: 600,
    color: '#374151',
    margin: '16px 0 8px',
  },
  table: {
    width: '100%',
    borderCollapse: 'collapse',
    fontSize: 12,
  },
  th: {
    textAlign: 'left',
    padding: '8px 10px',
    background: '#F9FAFB',
    color: '#6B7280',
    fontWeight: 500,
    borderBottom: '1px solid #E5E7EB',
  },
  td: {
    padding: '8px 10px',
    color: '#374151',
    borderBottom: '1px solid #F3F4F6',
  },
  trEven: {
    background: '#FAFAFA',
  },
  rateBadge: {
    display: 'inline-block',
    padding: '2px 8px',
    borderRadius: 20,
    fontSize: 11,
    fontWeight: 500,
  },
  cfBox: {
    background: '#F0FDF4',
    borderLeft: '3px solid #1D9E75',
    borderRadius: '0 6px 6px 0',
    padding: '8px 12px',
    margin: '6px 0',
    fontSize: 13,
    color: '#374151',
  },
  cfFeat: {
    fontWeight: 600,
    color: '#1D9E75',
  },
  cfNote: {
    fontSize: 12,
    color: '#6B7280',
    marginTop: 12,
    fontStyle: 'italic',
  },
}