// Polaris-style tinted status badges: light background + dark text per tone.
const ACTION_TONE = {
  RESTOCK: ['tone-critical', 'RESTOCK'],
  REVIEW_RESTOCK: ['tone-attention', 'REVIEW · RESTOCK?'],
  DISCOUNT: ['tone-info', 'CONSIDER DISCOUNT'],
  REVIEW_DISCOUNT: ['tone-attention', 'REVIEW · DISCOUNT?'],
  ON_TRACK: ['tone-success', 'ON TRACK'],
}

// The confidence-aware recommendation panel: the action, the plain-language message,
// and the numbers behind it (lead-time demand +/- sigma, safety stock, reorder point).
export default function PrescriptionCard({ presc }) {
  if (!presc) return <div className="card muted">Move a slider to get a recommendation…</div>
  const [tone, label] = ACTION_TONE[presc.action] ?? ['tone-neutral', presc.action]
  return (
    <div className="card presc">
      <div className="row spread">
        <h2>Recommended action</h2>
        <span className="chip">confidence: {presc.confidence}</span>
      </div>
      <div>
        <span className={`badge ${tone}`}>{label}</span>
      </div>
      <p style={{ margin: 0 }}>{presc.message}</p>
      <div className="metric-grid">
        <div className="metric">
          <div className="k">Expected sales over lead time</div>
          <div className="v">{presc.lead_time_demand} ± {presc.lead_time_demand_std}</div>
        </div>
        <div className="metric">
          <div className="k">Safety stock ({presc.service_level}% service)</div>
          <div className="v">{presc.safety_stock}</div>
        </div>
        <div className="metric">
          <div className="k">Reorder point</div>
          <div className="v">{presc.reorder_point}</div>
        </div>
        <div className="metric">
          <div className="k">Suggested order qty</div>
          <div className="v">{presc.suggested_order_qty}</div>
        </div>
        <div className="metric">
          <div className="k">Weeks of cover</div>
          <div className="v">{presc.weeks_of_cover_opt}</div>
        </div>
        <div className="metric">
          <div className="k">Discount effect on demand</div>
          <div className="v">×{presc.demand_factor}</div>
        </div>
      </div>
    </div>
  )
}
