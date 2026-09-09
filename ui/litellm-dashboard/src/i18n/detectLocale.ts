import { supportedLngs, type Locale } from "./resources/registry";

/**
 * Normalize a BCP-47 / browser language tag to one of the supported locales.
 *
 * Rules (I18N_TECH_DESIGN §3.1):
 *   - any Chinese tag (`zh`, `zh-Hans`, `zh-CN`, ...) -> `zh-CN`
 *   - an exact supported locale -> itself
 *   - anything else -> `en` (fallback)
 *
 * Pure function, safe to call in any environment.
 */
export function detectLocale(raw: string | null | undefined): Locale {
  if (!raw) return "en";

  const normalized = raw.trim().toLowerCase();

  if (normalized.startsWith("zh")) return "zh-CN";

  const exact = supportedLngs.find((lng) => lng.toLowerCase() === normalized);
  if (exact) return exact as Locale;

  return "en";
}

/** Accept a list of candidate browser languages and return the first supported one. */
export function detectLocaleFromList(candidates: readonly string[]): Locale {
  for (const candidate of candidates) {
    const detected = detectLocale(candidate);
    if (detected !== "en" || candidate.toLowerCase().startsWith("en")) {
      // A non-en detection is authoritative; for `en` we stop at the first en tag.
      return detected;
    }
  }
  return "en";
}

export function isSupportedLocale(value: string | null | undefined): value is Locale {
  return typeof value === "string" && (supportedLngs as readonly string[]).includes(value);
}
