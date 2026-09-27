// Tiny fetch wrapper: attaches the auth token, parses JSON, and turns backend
// {error: "..."} responses into thrown ApiError so pages can show the message.

const TOKEN_KEY = 'fyp_token'

export const getToken = () => localStorage.getItem(TOKEN_KEY)
export const setToken = (t) =>
  t ? localStorage.setItem(TOKEN_KEY, t) : localStorage.removeItem(TOKEN_KEY)

export class ApiError extends Error {
  constructor(message, status) {
    super(message)
    this.status = status
  }
}

export async function api(path, { method = 'GET', json, formData } = {}) {
  const headers = {}
  const token = getToken()
  if (token) headers['Authorization'] = `Bearer ${token}`
  if (json) headers['Content-Type'] = 'application/json'
  const resp = await fetch(path, {
    method,
    headers,
    body: json ? JSON.stringify(json) : formData,
  })
  let body = null
  try {
    body = await resp.json()
  } catch {
    /* empty or non-JSON body */
  }
  if (!resp.ok) throw new ApiError(body?.error || `HTTP ${resp.status}`, resp.status)
  return body
}
