import { createContext, useContext, useEffect, useState } from 'react'
import { api, getToken, setToken } from './api'

// Auth state lives here: the token in localStorage, the user object in context.
// On page load we restore the session by asking /api/me with the stored token.

const AuthCtx = createContext(null)
export const useAuth = () => useContext(AuthCtx)

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null)
  const [booting, setBooting] = useState(true)

  useEffect(() => {
    if (!getToken()) {
      setBooting(false)
      return
    }
    api('/api/me')
      .then(setUser)
      .catch(() => setToken(null)) // stale/expired token -> back to login
      .finally(() => setBooting(false))
  }, [])

  async function login(username, password) {
    const r = await api('/api/auth/login', { method: 'POST', json: { username, password } })
    setToken(r.token)
    setUser({ username: r.username, role: r.role })
  }

  async function register(username, password, role = 'staff', ownerCode = '') {
    await api('/api/auth/register', {
      method: 'POST',
      json: { username, password, role, admin_code: ownerCode },
    })
    await login(username, password)
  }

  function logout() {
    setToken(null)
    setUser(null)
  }

  return (
    <AuthCtx.Provider value={{ user, booting, login, register, logout }}>
      {children}
    </AuthCtx.Provider>
  )
}
