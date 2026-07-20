import { useState, useRef } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip,
  ResponsiveContainer, Cell
} from 'recharts'

const API_URL = 'http://localhost:8000'
const COLORS = ['#1D9E75', '#5DCAA5', '#9FE1CB']

// ── Agent Progress Bar ────────────────────────────────────────────────────────

function AgentProgress({ status, investigator, counterfactual, report }) {
  if (status === 'idle') return null

  const agents = [
    {
      name: 'Agent 1 — Investigator',
      icon: '🔍',
      done: !!investigator,
      active: status === 'running' && !investigator,
      detail: investigator
        ? `${investigator.total_failures} failures, top: ${investigator.top_features?.[0]?.feature}`
        : 'SHAP analysis running...',
    },
    {
      name: 'Agent 2 — Counterfactual',
      icon: '🔄',
      done: !!counterfactual && counterfactual.version !== 'placeholder',
      active: status === 'running' && !!investigator && !counterfactual,
      detail: counterfactual
        ? `${counterfactual.found}/${counterfactual.attempted} found (${(counterfactual.success_rate * 100).toFixed(0)}%)`
        : 'Waiting...',
    },
    {
      name: 'Agent 3 — Reporter',
      icon: '📄',
      done: !!report && report.version !== 'placeholder',
      active: false,
      detail: report?.version === 'placeholder'
        ? 'Coming in Week 5'
        : report ? 'Complete' : 'Waiting...',
    },
  ]

  return (
    <div style={styles.card}>
      <h3 style={styles.cardTitle}>Pipeline progress</h3>
      {agents.map((agent, i) => (
        <div key={i} style={styles.agentRow}>
          <div style={{
            ...styles.agentDot,
            background: agent.done ? '#1D9E75'
              : agent.active ? '#EF9F27'
                : '#E5E7EB',
          }} />
          <span style={styles.agentIcon}>{agent.icon}</span>
          <div style={{ flex: 1 }}>
            <div style={styles.agentName}>{agent.name}</div>
            <div style={styles.agentDetail}>{agent.detail}</div>
          </div>
          <span style={{
            fontSize: 12,
            color: agent.done ? '#1D9E75' : '#9CA3AF'
          }}>
            {agent.done ? '✓' : agent.active ? '⏳' : '○'}
          </span>
        </div>
      ))}
    </div>
  )
}

// ── Investigator Results ──────────────────────────────────────────────────────

function InvestigatorResults({ data }) {
  if (!data) return null

  const chartData = data.top_features.map(f => ({
    name: f.feature,
    shap: parseFloat(f.mean_abs_shap.toFixed(4)),
  }))

  return (
    <div style={styles.card}>
      <h3 style={styles.cardTitle}>🔍 Agent 1 — SHAP Investigation</h3>

      <div style={styles.statRow}>
        <div style={styles.stat}>
          <div style={styles.statNum}>{data.total_failures}</div>
          <div style={styles.statLabel}>Total failures</div>
        </div>
        <div style={styles.stat}>
          <div style={styles.statNum}>{data.top_features?.length}</div>
          <div style={styles.statLabel}>Top features</div>
        </div>
        <div style={styles.stat}>
          <div style={styles.statNum}>{data.failure_patterns?.length}</div>
          <div style={styles.statLabel}>Patterns found</div>
        </div>
      </div>

      <h4 style={styles.subTitle}>Top failure-driving features (mean |SHAP|)</h4>
      <ResponsiveContainer width="100%" height={160}>
        <BarChart data={chartData} layout="vertical"
          margin={{ left: 10, right: 20, top: 4, bottom: 4 }}>
          <XAxis type="number" tick={{ fontSize: 11 }} />
          <YAxis dataKey="name" type="category"
            tick={{ fontSize: 11 }} width={100} />
          <Tooltip formatter={(v) => v.toFixed(4)} />
          <Bar dataKey="shap" radius={[0, 4, 4, 0]}>
            {chartData.map((_, i) => (
              <Cell key={i} fill={COLORS[i % COLORS.length]} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>

      {data.failure_patterns?.length > 0 && (
        <>
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
                <tr key={i}>
                  <td style={styles.td}><code>{p.feature}</code></td>
                  <td style={styles.td}><code>{p.range}</code></td>
                  <td style={styles.td}>{p.failure_count}</td>
                  <td style={styles.td}>
                    <span style={{
                      ...styles.badge,
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
        </>
      )}
    </div>
  )
}

// ── Counterfactual Results ────────────────────────────────────────────────────

function CounterfactualResults({ data }) {
  if (!data || data.version === 'placeholder') return null
  if (data.found === 0) return (
    <div style={styles.card}>
      <h3 style={styles.cardTitle}>🔄 Agent 2 — Counterfactual Analysis</h3>
      <p style={{ fontSize: 13, color: '#6B7280' }}>
        No counterfactuals found in this set of mispredictions.
        This can happen with heavily imbalanced datasets.
      </p>
    </div>
  )

  const brittleness = data.avg_features_to_flip <= 1.5
    ? { label: '⚠️ Model is brittle', color: '#854F0B', bg: '#FAEEDA' }
    : { label: '✓ Model is moderately robust', color: '#085041', bg: '#E1F5EE' }

  return (
    <div style={styles.card}>
      <h3 style={styles.cardTitle}>🔄 Agent 2 — Counterfactual Analysis</h3>

      <div style={styles.statRow}>
        <div style={styles.stat}>
          <div style={styles.statNum}>{data.found}/{data.attempted}</div>
          <div style={styles.statLabel}>CFs found</div>
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

      <div style={{
        ...styles.badge,
        background: brittleness.bg,
        color: brittleness.color,
        padding: '6px 12px',
        display: 'inline-block',
        marginBottom: 12,
      }}>
        {brittleness.label} — avg {data.avg_features_to_flip} feature change(s) needed
      </div>

      <h4 style={styles.subTitle}>
        What would have fixed these predictions?
      </h4>
      {data.examples.slice(0, 5).map((ex, i) => {
        const feat = ex.features_changed?.[0]
        const delta = ex.delta?.[feat]
        const orig = ex.original_value?.[feat]
        const cfVal = ex.counterfactual_value?.[feat]
        const dir = delta > 0 ? 'increased' : 'decreased'

        return (
          <div key={i} style={styles.cfBox}>
            <span style={styles.cfFeat}>{feat}</span>{' '}
            {dir} by <strong>{Math.abs(delta).toFixed(2)}</strong>
            {' '}({ex.pct_change}% change,{' '}
            {orig?.toFixed(2)} → {cfVal?.toFixed(2)})
          </div>
        )
      })}
    </div>
  )
}

// ── Main App ──────────────────────────────────────────────────────────────────

export default function App() {
  const [modelFile, setModelFile] = useState(null)
  const [csvFile, setCsvFile] = useState(null)
  const [modelName, setModelName] = useState('')
  const [status, setStatus] = useState('idle')
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
      form.append('model_name', modelName || modelFile.name.replace('.pkl', ''))

      const res = await axios.post(`${API_URL}/analyze`, form)
      const { job_id } = res.data
      setStatus('running')

      pollRef.current = setInterval(() => pollResults(job_id), 3000)
      pollResults(job_id)

    } catch (e) {
      setStatus('failed')
      setErrorMsg(
        e.response?.data?.detail ||
        'Upload failed. Make sure the API is running on port 8000.')
    }
  }

  function handleReset() {
    stopPolling()
    setModelFile(null)
    setCsvFile(null)
    setModelName('')
    setStatus('idle')
    setResults(null)
    setErrorMsg('')
  }

  const isRunning = status === 'running' || status === 'uploading'

  return (
    <div style={styles.page}>
      <div style={styles.container}>

        {/* Header */}
        <div style={styles.header}>
          <h1 style={styles.title}>🔬 AI Autopsy</h1>
          <p style={styles.subtitle}>
            Autonomous ML failure investigation
          </p>
        </div>

        {/* Render free tier wake-up notice */}
        <div style={styles.notice}>
          ℹ️ First load may take 30s to wake the server (free hosting).
          Subsequent analyses are fast.
        </div>

        {/* Upload card */}
        <div style={styles.card}>
          <h3 style={styles.cardTitle}>Upload your model</h3>

          <label style={styles.label}>Model file (.pkl)</label>
          <input type="file" accept=".pkl" style={styles.fileInput}
            onChange={e => setModelFile(e.target.files[0])}
            disabled={isRunning} />
          {modelFile && <p style={styles.fileName}>✓ {modelFile.name}</p>}

          <label style={styles.label}>Mispredictions CSV</label>
          <input type="file" accept=".csv" style={styles.fileInput}
            onChange={e => setCsvFile(e.target.files[0])}
            disabled={isRunning} />
          {csvFile && <p style={styles.fileName}>✓ {csvFile.name}</p>}

          <label style={styles.label}>Model name (optional)</label>
          <input type="text" style={styles.textInput}
            placeholder="e.g. Credit Card Fraud Detector"
            value={modelName}
            onChange={e => setModelName(e.target.value)}
            disabled={isRunning} />

          <div style={styles.buttonRow}>
            <button
              style={{
                ...styles.button,
                opacity: (!isReady || isRunning) ? 0.5 : 1,
                cursor: (!isReady || isRunning) ? 'not-allowed' : 'pointer',
              }}
              onClick={handleSubmit}
              disabled={!isReady || isRunning}
            >
              {status === 'uploading' ? 'Uploading...'
                : status === 'running' ? 'Analysing...'
                  : 'Run Autopsy ▶'}
            </button>

            {(status === 'complete' || status === 'failed') && (
              <button style={styles.resetBtn} onClick={handleReset}>
                New analysis
              </button>
            )}
          </div>

          {status === 'running' && (
            <p style={styles.statusMsg}>
              ⏳ Pipeline running... auto-refreshing every 3s
            </p>
          )}
          {status === 'failed' && (
            <p style={{ ...styles.statusMsg, color: '#CC2222' }}>
              ❌ {errorMsg}
            </p>
          )}

          {results?.total_s && status === 'complete' && (
            <p style={styles.statusMsg}>
              ✓ Completed in {results.total_s}s
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
        <InvestigatorResults data={results?.investigator} />
        <CounterfactualResults data={results?.counterfactual} />

      </div>
    </div>
  )
}

// ── Styles ────────────────────────────────────────────────────────────────────

const styles = {
  page: {
    minHeight: '100vh',
    background: '#F3F4F6',
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
    padding: '24px 16px',
  },
  container: { maxWidth: 740, margin: '0 auto' },
  header: { marginBottom: 16 },
  title: { fontSize: 26, fontWeight: 700, color: '#111', margin: '0 0 4px' },
  subtitle: { fontSize: 14, color: '#6B7280', margin: 0 },
  notice: {
    background: '#EFF6FF',
    border: '1px solid #BFDBFE',
    borderRadius: 8,
    padding: '8px 14px',
    fontSize: 12,
    color: '#1E40AF',
    marginBottom: 14,
  },
  card: {
    background: '#fff',
    border: '1px solid #E5E7EB',
    borderRadius: 12,
    padding: '18px 22px',
    marginBottom: 14,
  },
  cardTitle: { fontSize: 15, fontWeight: 600, color: '#111', margin: '0 0 14px' },
  label: {
    display: 'block', fontSize: 13, fontWeight: 500, color: '#374151',
    marginBottom: 5, marginTop: 10
  },
  fileInput: { display: 'block', width: '100%', fontSize: 13, cursor: 'pointer' },
  textInput: {
    display: 'block', width: '100%', fontSize: 13, padding: '7px 10px',
    border: '1px solid #D1D5DB', borderRadius: 6, boxSizing: 'border-box',
    outline: 'none', color: '#111'
  },
  fileName: { fontSize: 12, color: '#1D9E75', margin: '3px 0 0' },
  buttonRow: { display: 'flex', gap: 10, marginTop: 18 },
  button: {
    background: '#1D9E75', color: '#fff', border: 'none',
    borderRadius: 8, padding: '10px 22px', fontSize: 14, fontWeight: 500,
  },
  resetBtn: {
    background: 'none', color: '#6B7280',
    border: '1px solid #D1D5DB', borderRadius: 8,
    padding: '10px 18px', fontSize: 14, cursor: 'pointer',
  },
  statusMsg: { fontSize: 13, color: '#6B7280', marginTop: 10 },
  agentRow: {
    display: 'flex', alignItems: 'center', gap: 10,
    padding: '8px 0', borderBottom: '1px solid #F3F4F6',
  },
  agentDot: { width: 8, height: 8, borderRadius: '50%', flexShrink: 0 },
  agentIcon: { fontSize: 16, flexShrink: 0 },
  agentName: { fontSize: 13, fontWeight: 500, color: '#111' },
  agentDetail: { fontSize: 11, color: '#9CA3AF', marginTop: 1 },
  statRow: {
    display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)',
    gap: 10, marginBottom: 16
  },
  stat: { background: '#F9FAFB', borderRadius: 8, padding: '10px', textAlign: 'center' },
  statNum: { fontSize: 20, fontWeight: 600, color: '#111' },
  statLabel: { fontSize: 11, color: '#6B7280', marginTop: 2 },
  subTitle: { fontSize: 13, fontWeight: 600, color: '#374151', margin: '14px 0 8px' },
  table: { width: '100%', borderCollapse: 'collapse', fontSize: 12 },
  th: {
    textAlign: 'left', padding: '7px 10px', background: '#F9FAFB',
    color: '#6B7280', fontWeight: 500, borderBottom: '1px solid #E5E7EB'
  },
  td: { padding: '7px 10px', color: '#374151', borderBottom: '1px solid #F3F4F6' },
  badge: {
    display: 'inline-block', padding: '2px 8px',
    borderRadius: 20, fontSize: 11, fontWeight: 500
  },
  cfBox: {
    background: '#F0FDF4', borderLeft: '3px solid #1D9E75',
    borderRadius: '0 6px 6px 0', padding: '8px 12px',
    margin: '5px 0', fontSize: 13, color: '#374151',
  },
  cfFeat: { fontWeight: 600, color: '#1D9E75' },
}