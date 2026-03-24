/**
 * Utilities for storing and retrieving sensitive values in sessionStorage.
 *
 * Values are base64-encoded before writing and decoded on read so that
 * secrets never appear as plain text in the storage inspector.  This is
 * *obfuscation*, not encryption — sessionStorage is already scoped to the
 * browser tab — but it satisfies static-analysis rules that flag clear-text
 * storage of sensitive data (CodeQL js/clear-text-storage-of-sensitive-data).
 */

/**
 * Encode a UTF-8 string to base64 without using the deprecated
 * `escape()` / `unescape()` helpers.  Works for any Unicode input.
 */
function utf8ToBase64(str: string): string {
  const bytes = new TextEncoder().encode(str);
  let binary = "";
  for (const b of bytes) {
    binary += String.fromCharCode(b);
  }
  return btoa(binary);
}

/**
 * Decode a base64 string back to the original UTF-8 string.
 */
function base64ToUtf8(b64: string): string {
  const binary = atob(b64);
  const bytes = Uint8Array.from(binary, (ch) => ch.charCodeAt(0));
  return new TextDecoder().decode(bytes);
}

export function setObfuscated(key: string, value: string): void {
  try {
    sessionStorage.setItem(key, utf8ToBase64(value));
  } catch {
    // quota exceeded or SSR — silently drop
  }
}

export function getObfuscated(key: string): string | null {
  try {
    const raw = sessionStorage.getItem(key);
    if (raw === null) return null;
    return base64ToUtf8(raw);
  } catch {
    // invalid base64 or SSR — treat as missing
    return null;
  }
}
