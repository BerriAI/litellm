import { jwtDecode } from "jwt-decode";

export function isJwtExpired(token: string): boolean {
  try {
    const decoded: any = jwtDecode(token);
    if (decoded && typeof decoded.exp === "number") {
      return decoded.exp * 1000 <= Date.now();
    }
    return false;
  } catch {
    // If we can't decode, treat as invalid/expired
    return true;
  }
}

export function decodeToken(token: string | null): Record<string, any> | null {
  if (!token) return null;
  try {
    return jwtDecode(token) as Record<string, any>;
  } catch {
    return null;
  }
}

export function checkTokenValidity(token: string | null): boolean {
  if (!token) return false;
  const decoded = decodeToken(token);
  return decoded !== null && !isJwtExpired(token);
}
