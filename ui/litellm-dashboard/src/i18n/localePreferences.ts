import { detectLocale, isSupportedLocale } from "./detectLocale";
import { type Locale } from "./resources/registry";

/**
 * Unified language preference storage key (DECISIONS.md P5).
 * Always read/write this key; feature agents must not touch storage directly.
 */
export const LOCALE_STORAGE_KEY = "litellm.locale";

export interface LocaleEnv {
  /** Raw `document.cookie` string, e.g. `"a=1; b=2"`. */
  getCookieString(): string;
  /**
   * Set a cookie. `secure` is decided by the environment so tests can assert the
   * attribute independently of the global `window`.
   */
  setCookie(name: string, value: string, opts: { secure: boolean }): void;
  removeCookie(name: string): void;
  storageGet(key: string): string | null;
  storageSet(key: string, value: string): void;
  storageRemove(key: string): void;
  browserLanguages(): readonly string[];
  /** Whether `Secure` should be attached to cookies (true in production over HTTPS). */
  isSecure(): boolean;
}

/** Convenience default for the browser environment. */
export function createBrowserLocaleEnv(): LocaleEnv {
  const inHttps =
    typeof window !== "undefined" && window.location.protocol === "https:";
  return {
    getCookieString: () => (typeof document === "undefined" ? "" : document.cookie),
    setCookie: (name, value, opts) => {
      if (typeof document === "undefined") return;
      document.cookie = `${name}=${encodeURIComponent(value)}; SameSite=Lax; path=/${opts.secure ? "; Secure" : ""}`;
    },
    removeCookie: (name) => {
      if (typeof document === "undefined") return;
      document.cookie = `${name}=; Max-Age=0; SameSite=Lax; path=/`;
    },
    storageGet: (key) => {
      try {
        return typeof localStorage === "undefined" ? null : localStorage.getItem(key);
      } catch {
        return null;
      }
    },
    storageSet: (key, value) => {
      try {
        localStorage?.setItem(key, value);
      } catch {
        /* storage may be unavailable (e.g. private mode); cookie still carries the value */
      }
    },
    storageRemove: (key) => {
      try {
        localStorage?.removeItem(key);
      } catch {
        /* best-effort */
      }
    },
    browserLanguages: () =>
      typeof navigator === "undefined" ? [] : Array.from(navigator.languages ?? [navigator.language]).filter(Boolean),
    isSecure: () => inHttps,
  };
}

/** Parse a raw cookie string and return the value for `name` or null. */
export function readCookie(rawCookies: string, name: string): string | null {
  if (!rawCookies) return null;
  for (const part of rawCookies.split(";")) {
    const idx = part.indexOf("=");
    if (idx === -1) continue;
    const key = part.slice(0, idx).trim();
    if (key === name) {
      try {
        return decodeURIComponent(part.slice(idx + 1).trim());
      } catch {
        return part.slice(idx + 1).trim();
      }
    }
  }
  return null;
}

/**
 * Read the explicitly stored locale. Per L1, if the unified key exists anywhere
 * (localStorage fast path first, then cookie) it is treated as the user's
 * explicit choice. Returns null when no preference has been written.
 */
export function readStoredLocale(env: LocaleEnv): Locale | null {
  const fromStorage = env.storageGet(LOCALE_STORAGE_KEY);
  const value = fromStorage ?? readCookie(env.getCookieString(), LOCALE_STORAGE_KEY);
  if (value === null || value === "") return null;
  return isSupportedLocale(value) ? value : null;
}

/** Persist an explicit user choice to both cookie and localStorage. */
export function writeLocalePreference(locale: Locale, env: LocaleEnv): void {
  env.storageSet(LOCALE_STORAGE_KEY, locale);
  env.setCookie(LOCALE_STORAGE_KEY, locale, { secure: env.isSecure() });
}

/** Remove the stored preference from both layers. */
export function clearLocalePreference(env: LocaleEnv): void {
  env.storageRemove(LOCALE_STORAGE_KEY);
  env.removeCookie(LOCALE_STORAGE_KEY);
}

/**
 * D5 priority resolution:
 *   1. explicit stored preference (cookie/localStorage)
 *   2. browser language (first supported, normalized via detectLocale)
 *   3. `en` (fallback)
 */
export function resolveLocale(env: LocaleEnv): Locale {
  const stored = readStoredLocale(env);
  if (stored) return stored;
  return detectLocaleFromBrowser(env) ?? "en";
}

function detectLocaleFromBrowser(env: LocaleEnv): Locale | null {
  const languages = env.browserLanguages();
  if (languages.length === 0) return null;

  // Filter by supportedLngs and take the first supported tag (D5: browser
  // languages are used only when no explicit preference exists).
  for (const raw of languages) {
    if (isEnglishTag(raw)) return "en";
    const detected = detectLocale(raw);
    if (detected !== "en") return detected; // e.g. zh -> zh-CN
  }
  return "en";
}

function isEnglishTag(raw: string): boolean {
  const normalized = raw.trim().toLowerCase();
  return normalized === "en" || normalized.startsWith("en-");
}
