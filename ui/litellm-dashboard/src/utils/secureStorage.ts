function encode(value: string): string {
  // btoa cannot handle characters outside Latin-1, so we percent-encode first.
  return btoa(unescape(encodeURIComponent(value)));
}

function decode(encoded: string): string {
  return decodeURIComponent(escape(atob(encoded)));
}

export function setSecureItem(key: string, value: string): void {
  try {
    window.sessionStorage.setItem(key, encode(value));
  } catch {
    // Storage full or unavailable — silently ignore.
  }
}

export function getSecureItem(key: string): string | null {
  try {
    const raw = window.sessionStorage.getItem(key);
    if (raw === null) return null;
    return decode(raw);
  } catch {
    // Corrupted or non-encoded legacy value — clear it.
    try {
      window.sessionStorage.removeItem(key);
    } catch {
      // ignore
    }
    return null;
  }
}
