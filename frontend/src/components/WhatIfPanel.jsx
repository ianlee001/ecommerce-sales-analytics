// Four what-if controls in a 2x2 grid (brief s8): demand side = discount (via the
// elasticity layer); inventory-policy side = lead time (capped at the forecast
// horizon), service level, current stock (defaults to the simulated trajectory).
// Copy is written for the shop owner - the maths lives in the thesis.
export default function WhatIfPanel({ whatIf, setWhatIf, leadMax, defaultInventory,
                                      factor, epsSource }) {
  const set = (k, v) => setWhatIf(w => ({ ...w, [k]: v }))
  return (
    <div className="whatif">
      <label className="field">
        <span>Discount: <span className="val">{whatIf.discount}%</span>
          {whatIf.discount > 0 && (
            <span className="muted"> → demand ×{factor.toFixed(2)}</span>
          )}
        </span>
        <input type="range" min="0" max="50" step="5" value={whatIf.discount}
               onChange={e => set('discount', +e.target.value)} />
        <span className="muted small">
          {epsSource === 'estimated'
            ? "price sensitivity learned from this product's sales history"
            : 'price sensitivity from a standard industry assumption'}
        </span>
      </label>

      <label className="field">
        <span>Lead time: <span className="val">{whatIf.lead} week{whatIf.lead > 1 ? 's' : ''}</span></span>
        <input type="range" min="1" max={leadMax} step="1" value={whatIf.lead}
               onChange={e => set('lead', +e.target.value)} />
        <span className="muted small">how long a new order takes to arrive (up to {leadMax} weeks)</span>
      </label>

      <div className="field">
        <span>Service level</span>
        <div className="seg">
          {[90, 95, 99].map(s => (
            <button key={s} className={whatIf.service === s ? 'on' : ''}
                    onClick={() => set('service', s)}>{s}%</button>
          ))}
        </div>
        <span className="muted small">higher service level = more safety stock, fewer stock-outs</span>
      </div>

      <label className="field">
        <span>Current stock</span>
        <div className="row" style={{ flexWrap: 'nowrap' }}>
          <input type="number" min="0" value={whatIf.inventory}
                 onChange={e => set('inventory', +e.target.value)} style={{ flex: 1 }} />
          {defaultInventory != null && (
            <button className="btn ghost small"
                    onClick={() => set('inventory', defaultInventory)}>
              use estimate ({Math.round(defaultInventory)})
            </button>
          )}
        </div>
      </label>
    </div>
  )
}
