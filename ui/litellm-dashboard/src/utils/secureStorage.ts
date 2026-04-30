function encode(value: string): string {
  // btoa cannot handle characters outside Latin-1, so we percent-encode first.
  return btoa(
    encodeURIComponent(value).replace(
      /%([0-9A-F]{2})/g,
      (_, p1) => String.fromCharCode(parseInt(p1, 16))
    )
  );
}

function decode(encoded: string): string {
  return decodeURIComponent(
    atob(encoded)
      .split("")
      .map((c) => "%" + c.charCodeAt(0).toString(16).padStart(2, "0"))
      .join("")
  );
}

export function setSecureItem(key: string, value: string): void {
  window.sessionStorage.setItem(key, encode(value));
}

export function getSecureItem(key: string): string | null {
  try {
    const raw = window.sessionStorage.getItem(key);
    if (raw === null) return null;
    return decode(raw);
  } catch {
    // Corrupted or non-encoded legacy value — return null without deleting
    // so that in-flight flows (e.g. OAuth) can time out naturally.
    return null;
  }
}
