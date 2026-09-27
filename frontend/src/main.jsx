import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import App from './App'
import { AuthProvider } from './auth'
import './index.css'

// NOTE: no <React.StrictMode> wrapper - StrictMode's dev-only double-mount interrupts
// recharts' entry animations (empty Pie sectors); production output is identical anyway.
ReactDOM.createRoot(document.getElementById('root')).render(
  <BrowserRouter>
    <AuthProvider>
      <App />
    </AuthProvider>
  </BrowserRouter>,
)
