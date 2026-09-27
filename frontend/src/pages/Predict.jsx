import { useEffect, useRef, useState } from 'react'
import { api } from '../api'
import ForecastChart from '../components/ForecastChart'
import PrescriptionCard from '../components/PrescriptionCard'
import WhatIfPanel from '../components/WhatIfPanel'

// The CORE page. One "Generate" = ONE MC-Dropout run on the backend; every slider
// change afterwards only re-prescribes closed-form from the cached mu/sigma numbers
// (and rescales the chart via the elasticity factor) - the model is never re-run.
//
// Two demo modes (told apart automatically by whether held-out actuals exist):
//   validation - the slice ends inside the held-out test period, so the horizon has
//                known actuals to overlay (green line): the ACCURACY EVIDENCE.
//   deployment - the slice ends at the last recorded week, so the forecast reaches a
//                genuine unknown future: what a real SME would see day-to-day.
export default function Predict() {
  const [examples, setExamples] = useState([])
  const [pick, setPick] = useState('')          // "mode:sku"
  const [file, setFile] = useState(null)
  const [result, setResult] = useState(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [whatIf, setWhatIf] = useState({ discount: 0, lead: 2, service: 95, inventory: 0 })
  const [presc, setPresc] = useState(null)
  const firstRun = useRef(true)   // the first inference also warms the model up

  useEffect(() => {
    api('/api/examples').then(setExamples).catch(e => setError(e.message))
  }, [])

  function applyResult(r) {
    firstRun.current = false
    setResult(r)
    setPresc(null)
    setWhatIf(w => ({
      ...w,
      discount: 0,
      inventory: r.default_inventory ?? w.inventory ?? 0,
      lead: Math.min(w.lead, r.lead_time_max),
    }))
  }

  async function generateFromFile() {
    setBusy(true); setError('')
    try {
      const fd = new FormData()
      fd.append('file', file)
      applyResult(await api('/api/forecast', { method: 'POST', formData: fd }))
    } catch (e) { setError(e.message) } finally { setBusy(false) }
  }

  async function generateFromExample() {
    setBusy(true); setError('')
    try {
      const [mode, sku] = pick.split(':')
      const rows = await api(`/api/example/${sku}?mode=${mode}`)
      applyResult(await api('/api/forecast', { method: 'POST', json: { rows } }))
    } catch (e) { setError(e.message) } finally { setBusy(false) }
  }

  // debounce slider changes -> closed-form /api/prescribe (raw arrays + slider values;
  // the backend applies the elasticity factor itself)
  useEffect(() => {
    if (!result) return
    const t = setTimeout(() => {
      api('/api/prescribe', {
        method: 'POST',
        json: {
          series_id: result.series_id,
          mu: result.forecast.map(f => f.mu),
          sigma: result.forecast.map(f => f.sigma),
          sigma_cum: result.forecast.map(f => f.sigma_cum),
          current_inventory: whatIf.inventory,
          lead_time: whatIf.lead,
          service_level: whatIf.service,
          discount_pct: whatIf.discount,
        },
      }).then(setPresc).catch(e => setError(e.message))
    }, 250)
    return () => clearTimeout(t)
  }, [result, whatIf])

  const factor = result && whatIf.discount > 0
    ? Math.pow(1 - whatIf.discount / 100, result.epsilon)
    : 1
  const isValidation = !!result?.forecast.some(f => f.actual != null)

  return (
    <>
      <div className="card">
        <h2>Sales prediction</h2>
        <p className="muted small">
          Upload a product's recent sales history (CSV, at least 12 weeks) to get a
          4-week forecast and a restock recommendation.
        </p>
        <div className="row">
          <input type="file" accept=".csv" onChange={e => setFile(e.target.files[0] ?? null)} />
          <button className="btn" disabled={!file || busy} onClick={generateFromFile}>
            {busy ? 'Forecasting…' : 'Upload & forecast'}
          </button>
        </div>
        <div className="row" style={{ marginTop: 8 }}>
          <span className="muted small">No file handy? Try a sample product: </span>
          <select value={pick} onChange={e => setPick(e.target.value)} style={{ width: 320 }}>
            <option value="">— choose a sample —</option>
            {examples.map(e => (
              <option key={`${e.mode}:${e.sku}`} value={`${e.mode}:${e.sku}`}>
                SKU {e.sku} — {e.mode === 'validation'
                  ? 'accuracy check (compare with real sales)'
                  : 'plan the next 4 weeks'}
              </option>
            ))}
          </select>
          <button className="btn ghost" disabled={!pick || busy} onClick={generateFromExample}>
            {busy ? 'Forecasting…' : 'Use sample'}
          </button>
        </div>
        {busy && (
          <div className="notice" style={{ marginTop: 10 }}>
            {firstRun.current
              ? 'Crunching the numbers — your first forecast can take up to 20 seconds…'
              : 'Updating the forecast…'}
          </div>
        )}
        {error && <div className="error" style={{ marginTop: 10 }}>{error}</div>}
      </div>

      {result && (
        <>
          <div className={`mode-banner ${isValidation ? 'validation' : 'deployment'}`}>
            {isValidation ? (
              <>
                <b>Accuracy check.</b> These 4 weeks already happened, but the model
                never saw them — the green line shows what really sold, so you can judge
                the forecast for yourself.
              </>
            ) : (
              <>
                <b>Planning ahead.</b> This forecast covers the next 4 weeks — beyond
                your recorded sales history (after
                {' '}{result.history[result.history.length - 1].date}).
              </>
            )}
          </div>
          {/* one-screen layout: chart + what-if stacked on the left, the
              recommendation always visible at the top right */}
          <div className="grid-2">
            <div style={{ display: 'grid', gap: 16 }}>
              <div className="card">
                <h2>SKU {result.series_id} — 4-week forecast</h2>
                <ForecastChart result={result} factor={factor} lead={whatIf.lead}
                               showActuals={isValidation} />
                <p className="muted small">
                  The red band is the likely range — about 8 weeks in 10 land inside it.
                  The shaded window is your delivery lead time.
                  {isValidation && ' Green = what actually sold.'}
                </p>
              </div>
              <div className="card">
                <h2>What-if</h2>
                <WhatIfPanel
                  whatIf={whatIf}
                  setWhatIf={setWhatIf}
                  leadMax={result.lead_time_max}
                  defaultInventory={result.default_inventory}
                  factor={factor}
                  epsSource={result.epsilon_source}
                />
                {result.default_inventory == null && (
                  <p className="muted small" style={{ marginBottom: 0 }}>
                    Enter your current stock to get an accurate recommendation.
                  </p>
                )}
              </div>
            </div>
            <PrescriptionCard presc={presc} />
          </div>
        </>
      )}
    </>
  )
}
