import { useEffect, useState } from 'react'
import { api } from '../api'
import { useAuth } from '../auth'

// One merged staff page (replaces the old Profile + Users pages): the first row is
// always yourself, then every colleague. Access control mirrors the backend:
//   admin  - change anyone's role, reset their password, delete them
//   anyone - change their OWN password (old password required)
export default function Staff() {
  const { user } = useAuth()
  const isAdmin = user.role === 'admin'
  const [rows, setRows] = useState([])
  const [error, setError] = useState('')
  const [open, setOpen] = useState(null)   // {id, action: 'own-pw' | 'reset-pw'}
  const [pw, setPw] = useState({ old: '', nw: '' })
  const [notice, setNotice] = useState('')

  const load = () => api('/api/users')
    .then(us => setRows([...us].sort((a, b) =>
      (a.username === user.username ? -1 : b.username === user.username ? 1 : a.id - b.id))))
    .catch(e => setError(e.message))
  useEffect(() => { load() }, [])

  const flash = (t) => { setNotice(t); setTimeout(() => setNotice(''), 3000) }
  const fail = (e) => setError(e.message)

  async function changeRole(u, role) {
    setError('')
    try { await api(`/api/users/${u.id}`, { method: 'PUT', json: { role } }); flash(`${u.username} is now ${role}`); load() }
    catch (e) { fail(e) }
  }
  async function remove(u) {
    if (!window.confirm(`Delete user "${u.username}"?`)) return
    setError('')
    try { await api(`/api/users/${u.id}`, { method: 'DELETE' }); flash(`${u.username} deleted`); load() }
    catch (e) { fail(e) }
  }
  async function submitPw(u) {
    setError('')
    try {
      if (open.action === 'own-pw') {
        await api('/api/auth/change-password', { method: 'POST',
          json: { old_password: pw.old, new_password: pw.nw } })
      } else {
        await api(`/api/users/${u.id}/password`, { method: 'POST',
          json: { new_password: pw.nw } })
      }
      flash('Password updated')
      setOpen(null); setPw({ old: '', nw: '' })
    } catch (e) { fail(e) }
  }

  const toggle = (id, action) => {
    setPw({ old: '', nw: '' })
    setOpen(o => (o && o.id === id && o.action === action ? null : { id, action }))
  }

  return (
    <div className="card">
      <h2>Staff</h2>
      <p className="muted small">
        {isAdmin
          ? 'You are the owner: you can change roles, reset passwords and remove accounts.'
          : 'You can change your own password; account management is owner-only.'}
      </p>
      {error && <div className="error">{error}</div>}
      {notice && <div className="notice">{notice}</div>}
      <table>
        <thead>
          <tr><th>User</th><th>Role</th><th>Created</th><th style={{ width: 320 }}>Actions</th></tr>
        </thead>
        <tbody>
          {rows.map(u => {
            const me = u.username === user.username
            return (
              <FragmentRow key={u.id} u={u} me={me} isAdmin={isAdmin} open={open}
                           toggle={toggle} changeRole={changeRole} remove={remove}
                           pw={pw} setPw={setPw} submitPw={submitPw} />
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

function FragmentRow({ u, me, isAdmin, open, toggle, changeRole, remove, pw, setPw, submitPw }) {
  const opened = open && open.id === u.id
  return (
    <>
      <tr>
        <td>{u.username}{me && <span className="chip" style={{ marginLeft: 6 }}>you</span>}</td>
        <td>
          {isAdmin && !me ? (
            <select value={u.role} onChange={e => changeRole(u, e.target.value)} style={{ width: 100 }}>
              <option value="staff">staff</option>
              <option value="admin">admin</option>
            </select>
          ) : u.role}
        </td>
        <td className="muted">{u.created_at}</td>
        <td>
          <div className="row">
            {me && (
              <button className="btn ghost small" onClick={() => toggle(u.id, 'own-pw')}>
                Change password
              </button>
            )}
            {isAdmin && !me && (
              <>
                <button className="btn ghost small" onClick={() => toggle(u.id, 'reset-pw')}>
                  Reset password
                </button>
                <button className="btn danger small" onClick={() => remove(u)}>Delete</button>
              </>
            )}
          </div>
        </td>
      </tr>
      {opened && (
        <tr>
          <td colSpan={4} style={{ background: '#fafbff' }}>
            <div className="row">
              {open.action === 'own-pw' && (
                <input type="password" placeholder="current password" value={pw.old}
                       onChange={e => setPw(p => ({ ...p, old: e.target.value }))}
                       style={{ width: 200 }} />
              )}
              <input type="password" placeholder="new password (≥ 6 chars)" value={pw.nw}
                     onChange={e => setPw(p => ({ ...p, nw: e.target.value }))}
                     style={{ width: 200 }} />
              <button className="btn small" disabled={pw.nw.length < 6}
                      onClick={() => submitPw(u)}>Save</button>
            </div>
          </td>
        </tr>
      )}
    </>
  )
}
