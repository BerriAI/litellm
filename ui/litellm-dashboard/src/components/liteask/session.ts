import { jwtDecode } from "jwt-decode";

interface SessionClaims {
  user_role?: unknown;
  user_id?: unknown;
  key?: unknown;
  exp?: unknown;
  password_reset_required?: unknown;
}

interface AuthSession {
  token: string | null;
  accessToken: string | null;
  userID: string | null;
  authLoading: boolean;
  passwordResetRequired: boolean;
}

export function getLiteAskSession(auth: AuthSession, now = Date.now()): { expiresAt: number | null } | null {
  const authenticationBlocked = auth.authLoading || auth.passwordResetRequired;
  if (authenticationBlocked || !auth.token) return null;
  if (!auth.accessToken || !auth.userID) return null;
  try {
    const claims = jwtDecode<SessionClaims>(auth.token);
    if (claims.user_role !== "proxy_admin" || claims.password_reset_required === true) return null;
    if (claims.user_id !== auth.userID || claims.key !== auth.accessToken) return null;
    if (claims.exp === undefined) return { expiresAt: null };
    if (typeof claims.exp !== "number" || !Number.isFinite(claims.exp) || claims.exp * 1000 <= now) return null;
    return { expiresAt: claims.exp * 1000 };
  } catch {
    return null;
  }
}
