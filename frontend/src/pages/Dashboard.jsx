import { useEffect, useMemo, useState } from 'react'
import {
  Bar, BarChart, CartesianGrid, Cell, Legend, Line, LineChart, Pie, PieChart,
  ReferenceArea, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis,
} from 'recharts'
import { api } from '../api'

const COLORS = ['#3a7bd5', '#d5573a', '#2a9d4a', '#e8a33d', '#7d6fb0',
                '#4fb3bf', '#c05299', '#8d99ae']
const TABS = ['Overview', 'Analysis', 'Product Table', 'Price Insight']

// Count-up hook: animates a number from 0 to `target` (ease-out) when it first appears.
function useCountUp(target, ms = 900) {
  const [v, setV] = useState(0)
  useEffect(() => {
    if (target == null) return
    let raf, t0
    const step = (t) => {
      if (!t0) t0 = t
      const p = Math.min((t - t0) / ms, 1)
      setV(target * (1 - Math.pow(1 - p, 3)))
      if (p < 1) raf = requestAnimationFrame(step)
    }
    raf = requestAnimationFrame(step)
    return () => cancelAnimationFrame(raf)
  }, [target, ms])
  return v
}

function StatCard({ label, value, prefix = '', text }) {
  const n = useCountUp(text ? null : value)
  return (
    <div className="stat-card">
      <div className="k">{label}</div>
      <div className="v">{text ?? `${prefix}${Math.round(n).toLocaleString()}`}</div>
    </div>
  )
}

function Overview({ data }) {
  const t = data.business.totals
  // exact category values for the Q4 shading (category XAxis needs real tick values)
  const q4spans = useMemo(() => {
    const dates = data.total_weekly.map(w => w.date)
    const spans = []
    for (const y of [2009, 2010, 2011]) {
      const inQ4 = dates.filter(d => d >= `${y}-10-01` && d <= `${y}-12-31`)
      if (inQ4.length > 1) spans.push([inQ4[0], inQ4[inQ4.length - 1]])
    }
    return spans
  }, [data])

  return (
    <>
      <div className="stat-grid">
        <StatCard label="Total revenue" value={t.revenue} prefix="£" />
        <StatCard label="Units sold" value={t.units} />
        <StatCard label="Orders" value={t.invoices} />
        <StatCard label="Products" value={t.products} />
        <StatCard label="Customers" value={t.customers} />
        <StatCard label="Main market" text={t.main_market} />
      </div>
      <div className="card">
        <h2>Total weekly sales — {data.date_range[0]} to {data.date_range[1]}</h2>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={data.total_weekly} margin={{ top: 8, right: 12, bottom: 0, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#eef0f6" />
            <XAxis dataKey="date" tickFormatter={d => d.slice(0, 7)} minTickGap={40} fontSize={11} />
            <YAxis fontSize={11} width={64} />
            <Tooltip labelStyle={{ fontWeight: 700 }} formatter={v => [v.toLocaleString(), 'units']} />
            {q4spans.map(([a, b]) => (
              <ReferenceArea key={a} x1={a} x2={b} fill="#e8a33d" fillOpacity={0.12}
                label={{ value: 'Q4', position: 'insideTop', fontSize: 11, fill: '#b97d16' }} />
            ))}
            <Line type="monotone" dataKey="sales" name="units / week" stroke="#3a7bd5"
                  strokeWidth={1.8} dot={false} animationDuration={1400} />
          </LineChart>
        </ResponsiveContainer>
        <p className="muted small">
          Sales climb every autumn and peak in the run-up to Christmas (shaded).
          Hover any point to see that week's number.
        </p>
      </div>
    </>
  )
}

function Analysis({ data }) {
  const months = useMemo(() =>
    [...data.business.best_months].sort((a, b) => b.revenue - a.revenue), [data])
  const markets = data.business.markets
  const total = markets.reduce((s, m) => s + m.revenue, 0)
  return (
    <div className="grid-2">
      <div className="card">
        <h2>Revenue by market</h2>
        <div className="row" style={{ alignItems: 'center', flexWrap: 'nowrap' }}>
          {/* Only the PIE spins in. recharts' own pie animation is broken in this
              version, so the sweep is a CSS entrance - and the legend must live
              OUTSIDE the animated wrapper so it stays perfectly still. */}
          <div className="pie-enter" style={{ flex: '1 1 62%' }}>
            <ResponsiveContainer width="100%" height={280}>
              <PieChart>
                <Pie data={markets} dataKey="revenue" nameKey="name"
                     cx="50%" cy="50%" outerRadius={100} isAnimationActive={false}>
                  {markets.map((m, i) => (
                    <Cell key={m.name} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip formatter={v => `£${Math.round(v).toLocaleString()}`} />
              </PieChart>
            </ResponsiveContainer>
          </div>
          <ul className="pie-legend">
            {markets.map((m, i) => (
              <li key={m.name}>
                <span className="dot" style={{ background: COLORS[i % COLORS.length] }} />
                {m.name} <b>{(m.revenue / total * 100).toFixed(1)}%</b>
              </li>
            ))}
          </ul>
        </div>
        <p className="muted small">
          Where the revenue comes from — mostly the UK, with steady overseas wholesale.
        </p>
      </div>
      <div className="card">
        <h2>Best months by revenue</h2>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={months} layout="vertical" margin={{ top: 4, right: 16, bottom: 0, left: 8 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#eef0f6" horizontal={false} />
            <XAxis type="number" fontSize={11}
                   tickFormatter={v => `£${(v / 1e6).toFixed(1)}M`} />
            <YAxis type="category" dataKey="month" fontSize={11} width={36} />
            <Tooltip formatter={v => `£${Math.round(v).toLocaleString()}`} />
            <Bar dataKey="revenue" fill="#3a7bd5" animationDuration={1100} radius={[0, 4, 4, 0]} />
          </BarChart>
        </ResponsiveContainer>
        <p className="muted small">
          October to December are the biggest months — get stock in before the rush.
        </p>
      </div>
    </div>
  )
}

const SORT_OPTIONS = [
  ['revenue', 'Revenue'],
  ['units', 'Units sold'],
  ['avg_price', 'Avg price'],
  ['sim_stock', 'Stock'],
  ['est_value', 'Stock value'],
]

function ProductTable({ data }) {
  const [sort, setSort] = useState({ key: 'revenue', dir: -1 })
  const [q, setQ] = useState('')
  const rows = useMemo(() => {
    const needle = q.trim().toLowerCase()
    return data.product_table
      .filter(r => !needle
        || r.series_id.toLowerCase().includes(needle)
        || r.description.toLowerCase().includes(needle))
      .sort((a, b) =>
        (a[sort.key] > b[sort.key] ? 1 : a[sort.key] < b[sort.key] ? -1 : 0) * sort.dir)
  }, [data, sort, q])
  const th = (key, label) => (
    <th className="sortable" onClick={() =>
        setSort(s => ({ key, dir: s.key === key ? -s.dir : -1 }))}>
      {label}{sort.key === key ? (sort.dir === -1 ? ' ↓' : ' ↑') : ''}
    </th>
  )
  function exportCsv() {
    const cols = ['series_id', 'description', 'units', 'revenue', 'avg_price', 'sim_stock', 'est_value']
    const lines = [cols.join(','), ...rows.map(r =>
      cols.map(c => `"${String(r[c]).replace(/"/g, '""')}"`).join(','))]
    const a = document.createElement('a')
    a.href = URL.createObjectURL(new Blob([lines.join('\n')], { type: 'text/csv' }))
    a.download = 'product_table.csv'
    a.click()
    URL.revokeObjectURL(a.href)
  }
  return (
    <div className="card">
      <div className="row spread">
        <h2>Top products ({rows.length})</h2>
        <div className="row">
          <input type="text" placeholder="Search stock code or description…"
                 value={q} onChange={e => setQ(e.target.value)} style={{ width: 250 }} />
          <span className="muted small">Sort by</span>
          <select value={sort.key}
                  onChange={e => setSort({ key: e.target.value, dir: -1 })}>
            {SORT_OPTIONS.map(([k, label]) => (
              <option key={k} value={k}>{label}</option>
            ))}
          </select>
          <button className="btn ghost small" onClick={exportCsv}>Export CSV</button>
        </div>
      </div>
      <table>
        <thead>
          <tr>
            <th>SKU</th><th>Description</th>
            {th('units', 'Units sold')}
            {th('revenue', 'Revenue (£)')}
            {th('avg_price', 'Avg price')}
            {th('sim_stock', 'Stock*')}
            {th('est_value', 'Stock value* (£)')}
          </tr>
        </thead>
        <tbody>
          {rows.length === 0 && (
            <tr><td colSpan={7} className="muted">No products match "{q}".</td></tr>
          )}
          {rows.map(r => (
            <tr key={r.series_id}>
              <td>{r.series_id}</td>
              <td>{r.description}</td>
              <td>{r.units.toLocaleString()}</td>
              <td>{Math.round(r.revenue).toLocaleString()}</td>
              <td>{r.avg_price}</td>
              <td>{r.sim_stock.toLocaleString()}</td>
              <td>{Math.round(r.est_value).toLocaleString()}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="muted small">
        * Stock and stock value are simulated for this demo — the sales history contains
        no stock records.
      </p>
    </div>
  )
}

function PriceInsight({ data }) {
  // "what would a 20% discount do?" - the same elasticity formula the Prediction
  // page's slider applies, shown per SKU so discount candidates are obvious
  const DISCOUNT = 20
  const uplift = useMemo(() =>
    data.price_insight.discount_readiness
      .map(r => ({ ...r, uplift: (Math.pow(1 - DISCOUNT / 100, r.epsilon) - 1) * 100 }))
      .sort((a, b) => b.uplift - a.uplift),
    [data])
  return (
    <div className="grid-2">
      <div className="card">
        <h2>Price vs weekly sales</h2>
        <ResponsiveContainer width="100%" height={300}>
          <ScatterChart margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#eef0f6" />
            <XAxis type="number" dataKey="avg_price" name="avg price" unit=" £" fontSize={11} />
            <YAxis type="number" dataKey="avg_weekly_sales" name="avg weekly sales" fontSize={11} width={56} />
            <Tooltip cursor={{ strokeDasharray: '3 3' }}
                     formatter={(v, n) => [v, n]}
                     labelFormatter={() => ''}
                     content={({ payload }) => payload?.length ? (
                       <div className="card small" style={{ padding: 8 }}>
                         <b>SKU {payload[0].payload.series_id}</b><br />
                         avg price £{payload[0].payload.avg_price}<br />
                         avg {payload[0].payload.avg_weekly_sales} units/week
                       </div>
                     ) : null} />
            <Scatter data={data.price_insight.scatter} fill="#3a7bd5" animationDuration={900} />
          </ScatterChart>
        </ResponsiveContainer>
        <p className="muted small">
          Cheaper items sell in larger weekly volumes.
        </p>
      </div>
      <div className="card">
        <h2>Expected demand uplift at {DISCOUNT}% off</h2>
        <ResponsiveContainer width="100%" height={440}>
          <BarChart data={uplift} layout="vertical"
                    margin={{ top: 4, right: 36, bottom: 0, left: 8 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#eef0f6" horizontal={false} />
            <XAxis type="number" fontSize={11} unit="%" />
            <YAxis type="category" dataKey="series_id" fontSize={11} width={52} />
            <Tooltip content={({ payload }) => payload?.length ? (
              <div className="card small" style={{ padding: 8 }}>
                <b>SKU {payload[0].payload.series_id}</b><br />
                about +{payload[0].payload.uplift.toFixed(0)}% demand at {DISCOUNT}% off<br />
                {payload[0].payload.source === 'estimated'
                  ? "based on this product's sales history"
                  : 'based on a standard industry assumption'}
              </div>
            ) : null} />
            <Bar dataKey="uplift" animationDuration={900} radius={[0, 4, 4, 0]}>
              {uplift.map(r => (
                <Cell key={r.series_id}
                      fill={r.source === 'estimated' ? '#3a7bd5' : '#b9c2d0'} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
        <p className="muted small">
          Estimated extra demand if you run a 20% discount — the same maths behind the
          discount slider on the Prediction page.
          <b style={{ color: '#3a7bd5' }}> Blue</b> = learned from this shop's own sales;
          <b style={{ color: '#8d99ae' }}> grey</b> = standard industry assumption.
          The longest blue bars are your best discount candidates.
        </p>
      </div>
    </div>
  )
}

export default function Dashboard() {
  const [data, setData] = useState(null)
  const [error, setError] = useState('')
  const [tab, setTab] = useState('Overview')
  const [dir, setDir] = useState('right')   // which side the incoming panel slides from

  useEffect(() => {
    api('/api/eda/dashboard').then(setData).catch(e => setError(e.message))
  }, [])

  // directional slide (Polaris: motion communicates the tabs' spatial order):
  // moving to a tab further right slides in from the right, and vice versa
  function switchTab(next) {
    if (next === tab) return
    setDir(TABS.indexOf(next) > TABS.indexOf(tab) ? 'right' : 'left')
    setTab(next)
  }

  if (error) return <div className="error">{error}</div>
  if (!data) return <div className="card muted">Loading dashboard…</div>

  return (
    <>
      <div className="tabs">
        {TABS.map(t => (
          <button key={t} className={tab === t ? 'on' : ''} onClick={() => switchTab(t)}>{t}</button>
        ))}
      </div>
      {/* key={tab} remounts the panel so the slide (and chart entrances) replay */}
      <div key={tab} className={`tab-panel slide-${dir}`}>
        {tab === 'Overview' && <Overview data={data} />}
        {tab === 'Analysis' && <Analysis data={data} />}
        {tab === 'Product Table' && <ProductTable data={data} />}
        {tab === 'Price Insight' && <PriceInsight data={data} />}
      </div>
    </>
  )
}
