import { useEffect, useState } from 'react'
import { api } from '../api'
import { useAuth } from '../auth'

// The prescriptive loop's human hand-off: the owner sees a RESTOCK alert on the
// prediction page and tells an employee to act (or broadcasts to everyone). Employees
// read messages addressed to them plus broadcasts. Deliberately thin - no threads.
export default function Messages() {
  const { user } = useAuth()
  const isAdmin = user.role === 'admin'
  const [msgs, setMsgs] = useState([])
  const [staff, setStaff] = useState([])
  const [to, setTo] = useState('')          // '' = everyone
  const [body, setBody] = useState('')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)

  const load = () => api('/api/messages').then(setMsgs).catch(e => setError(e.message))
  useEffect(() => {
    load()
    api('/api/users').then(setStaff).catch(() => {})
  }, [])

  async function send() {
    setBusy(true); setError('')
    try {
      await api('/api/messages', { method: 'POST', json: { to: to || null, body } })
      setBody('')
      load()
    } catch (e) { setError(e.message) } finally { setBusy(false) }
  }

  return (
    <>
      {isAdmin && (
        <div className="card">
          <h2>New message</h2>
          <div className="row">
            <select value={to} onChange={e => setTo(e.target.value)} style={{ width: 220 }}>
              <option value="">Everyone (broadcast)</option>
              {staff.filter(s => s.username !== user.username).map(s => (
                <option key={s.id} value={s.username}>{s.username}</option>
              ))}
            </select>
            <input type="text" placeholder="e.g. SKU 22411 is below the reorder point - please order 700 units"
                   value={body} onChange={e => setBody(e.target.value)} style={{ flex: 1 }}
                   maxLength={500}
                   onKeyDown={e => { if (e.key === 'Enter' && body.trim()) send() }} />
            <button className="btn" disabled={busy || !body.trim()} onClick={send}>Send</button>
          </div>
        </div>
      )}
      <div className="card">
        <h2>Messages</h2>
        {error && <div className="error">{error}</div>}
        {msgs.length === 0 && <p className="muted">No messages yet.</p>}
        <div className="msg-list">
          {msgs.map(m => (
            <div key={m.id} className="msg">
              <div className="row spread">
                <span>
                  <b>{m.from_user}</b>
                  <span className="muted"> → </span>
                  {m.to_user
                    ? <b>{m.to_user === user.username ? 'you' : m.to_user}</b>
                    : <span className="badge tone-info">ALL</span>}
                </span>
                <span className="muted small">{m.created_at}</span>
              </div>
              <p style={{ margin: '6px 0 0' }}>{m.body}</p>
            </div>
          ))}
        </div>
      </div>
    </>
  )
}
