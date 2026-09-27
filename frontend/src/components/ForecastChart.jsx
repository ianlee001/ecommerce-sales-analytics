import { useMemo } from 'react'
import {
  Area, CartesianGrid, ComposedChart, Legend, Line, ReferenceArea,
  ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts'

// History line + forecast mean + P10-P90 range band + held-out actuals + lead-time
// window. `factor` is the elasticity demand multiplier from the discount slider - the
// whole forecast (mean and band) scales with it, matching the backend formula.
export default function ForecastChart({ result, factor, lead, showActuals = true }) {
  const data = useMemo(() => {
    const hist = result.history.map(h => ({ date: h.date, sales: h.sales }))
    const last = hist[hist.length - 1]
    // bridge row so the dashed mean and the green actual line connect to the history
    const bridge = { ...last, mu: last.sales, band: [last.sales, last.sales], actual: last.sales }
    const fc = result.forecast.map(f => ({
      date: f.date,
      mu: +(f.mu * factor).toFixed(1),
      band: [+Math.max(0, f.p10 * factor).toFixed(1), +(f.p90 * factor).toFixed(1)],
      actual: f.actual,
    }))
    return [...hist.slice(0, -1), bridge, ...fc]
  }, [result, factor])

  const fcDates = result.forecast.map(f => f.date)
  // the shaded lead-time window starts at "now" (the last observed week), so lead = 1
  // still paints a visible band up to week 1 instead of a zero-width area
  const boundary = result.history[result.history.length - 1].date
  const leadEnd = fcDates[Math.min(lead, fcDates.length) - 1]

  return (
    <ResponsiveContainer width="100%" height={340}>
      <ComposedChart data={data} margin={{ top: 8, right: 12, bottom: 0, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#eef0f6" />
        <XAxis dataKey="date" tickFormatter={d => d.slice(5)} minTickGap={28} fontSize={11} />
        <YAxis fontSize={11} width={48} />
        <Tooltip
          formatter={(v, name) => (Array.isArray(v) ? `${v[0]} – ${v[1]}` : v)}
          labelStyle={{ fontWeight: 700 }}
        />
        <Legend wrapperStyle={{ fontSize: 12 }} />
        <ReferenceArea
          x1={boundary} x2={leadEnd}
          fill="#3a7bd5" fillOpacity={0.07}
          label={{ value: 'lead time', position: 'insideTopRight', fontSize: 11, fill: '#3a7bd5' }}
        />
        <Area type="monotone" dataKey="band" name="P10–P90" stroke="none"
              fill="#d5573a" fillOpacity={0.15} isAnimationActive={false} />
        <Line type="monotone" dataKey="sales" name="history" stroke="#3a7bd5"
              strokeWidth={1.6} dot={false} isAnimationActive={false} />
        <Line type="monotone" dataKey="mu" name="forecast mean" stroke="#d5573a"
              strokeWidth={1.8} strokeDasharray="6 3" dot={{ r: 3 }} isAnimationActive={false} />
        {showActuals && (
          <Line type="monotone" dataKey="actual" name="actual (held-out)" stroke="#2a9d4a"
                strokeWidth={1.6} dot={{ r: 3 }} isAnimationActive={false} />
        )}
      </ComposedChart>
    </ResponsiveContainer>
  )
}
