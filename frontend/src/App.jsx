import { NavLink, Navigate, Route, Routes, useNavigate } from 'react-router-dom'
import { useAuth } from './auth'
import Login from './pages/Login'
import Dashboard from './pages/Dashboard'
import Predict from './pages/Predict'
import Staff from './pages/Staff'
import Messages from './pages/Messages'

function Sidebar() {
  const { user, logout } = useAuth()
  const navigate = useNavigate()
  return (
    <aside className="sidebar">
      <div className="brand">UK Giftware<br />Analytics</div>
      <nav>
        <NavLink to="/" end>Dashboard</NavLink>
        <NavLink to="/predict">Prediction</NavLink>
        <NavLink to="/staff">Staff</NavLink>
        <NavLink to="/messages">Messages</NavLink>
      </nav>
      <div className="sidebar-foot">
        <div className="who">{user.username}<br /><span className="chip">{user.role}</span></div>
        <button className="btn ghost small" onClick={() => { logout(); navigate('/') }}>
          Log out
        </button>
      </div>
    </aside>
  )
}

export default function App() {
  const { user, booting } = useAuth()
  if (booting) return <div className="center-page">Loading…</div>
  if (!user) {
    return (
      <Routes>
        <Route path="*" element={<Login />} />
      </Routes>
    )
  }
  return (
    <div className="shell">
      <Sidebar />
      <main className="page">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/predict" element={<Predict />} />
          <Route path="/staff" element={<Staff />} />
          <Route path="/messages" element={<Messages />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>
    </div>
  )
}
