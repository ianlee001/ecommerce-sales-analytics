import { useState } from 'react'
import { useAuth } from '../auth'

export default function Login() {
  const { login, register } = useAuth()
  const [mode, setMode] = useState('login')
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [role, setRole] = useState('staff')
  const [ownerCode, setOwnerCode] = useState('')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)

  async function submit(e) {
    e.preventDefault()
    setBusy(true)
    setError('')
    try {
      if (mode === 'login') await login(username, password)
      else await register(username, password, role, ownerCode)
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy(false)
    }
  }

  const canSubmit = username && password && (mode === 'login' || role !== 'admin' || ownerCode)

  return (
    <div className="login-wrap">
      <div className="login-panel">
        <div className="login-hero">
          <h1>UK Giftware<br />Analytics</h1>
          <p className="tag">Know next month's sales.<br />Act on them today.</p>
          <ul>
            <li>See the next 4 weeks of sales before they happen</li>
            <li>Clear advice: when to restock, when to discount</li>
            <li>Try what-ifs: discounts, delivery times, service levels</li>
          </ul>
        </div>
        <form className="login-card" onSubmit={submit}>
          <div className="login-tabs">
            <button type="button" className={mode === 'login' ? 'on' : ''}
                    onClick={() => setMode('login')}>Log in</button>
            <button type="button" className={mode === 'register' ? 'on' : ''}
                    onClick={() => setMode('register')}>Register</button>
          </div>
          <label className="field">Username
            <input type="text" value={username} onChange={e => setUsername(e.target.value)}
                   autoFocus autoComplete="username" placeholder="e.g. shopowner" />
          </label>
          <label className="field">Password
            <input type="password" value={password} onChange={e => setPassword(e.target.value)}
                   autoComplete={mode === 'login' ? 'current-password' : 'new-password'}
                   placeholder={mode === 'register' ? 'at least 6 characters' : ''} />
          </label>
          {mode === 'register' && (
            <>
              <div className="field">
                <span>Role</span>
                <div className="seg">
                  <button type="button" className={role === 'staff' ? 'on' : ''}
                          onClick={() => setRole('staff')}>Staff</button>
                  <button type="button" className={role === 'admin' ? 'on' : ''}
                          onClick={() => setRole('admin')}>Owner (admin)</button>
                </div>
              </div>
              {role === 'admin' && (
                <label className="field">Owner code
                  <input type="text" value={ownerCode} onChange={e => setOwnerCode(e.target.value)}
                         placeholder="ask the system owner" />
                  <span className="muted small">
                    Owner accounts manage staff and broadcast messages, so registering one
                    needs the invite code (demo default: FYP-OWNER).
                  </span>
                </label>
              )}
            </>
          )}
          {error && <div className="error">{error}</div>}
          <button className="btn" disabled={busy || !canSubmit}>
            {busy ? 'Please wait…' : mode === 'login' ? 'Log in' : 'Create account'}
          </button>
          {mode === 'login' && (
            <span className="muted small" style={{ textAlign: 'center' }}>
              Demo login: admin / admin123
            </span>
          )}
        </form>
      </div>
    </div>
  )
}
