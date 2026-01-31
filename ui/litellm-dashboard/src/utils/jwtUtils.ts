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
