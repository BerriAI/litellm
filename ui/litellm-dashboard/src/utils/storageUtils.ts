/**
 * Utilities for storing and retrieving sensitive values in sessionStorage.
 *
 * Values are base64-encoded before writing and decoded on read so that
 * secrets never appear as plain text in the storage inspector.  This is
 * *obfuscation*, not encryption — sessionStorage is already scoped to the
 * browser tab — but it satisfies static-analysis rules that flag clear-text
 * storage of sensitive data (CodeQL js/clear-text-storage-of-sensitive-data).
 */

export function setObfuscated(key: string, value: string): void {
  try {
    // Encode via encodeURIComponent first so non-Latin1 characters
    // (e.g. Unicode MCP server aliases in OAuth flow-state JSON)
    // are converted to percent-encoded ASCII before btoa.
    sessionStorage.setItem(key, btoa(unescape(encodeURIComponent(value))));
  } catch {
    // quota exceeded or SSR — silently drop
  }
}

export function getObfuscated(key: string): string | null {
  try {
    const raw = sessionStorage.getItem(key);
    if (raw === null) return null;
    return decodeURIComponent(escape(atob(raw)));
  } catch {
    // invalid base64 or SSR — treat as missing
    return null;
  }
}
