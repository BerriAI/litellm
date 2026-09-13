"use client";

import { createContext, useCallback, useContext, useEffect, useState, type ReactNode } from "react";
import { I18nextProvider } from "react-i18next";
import type { i18n } from "i18next";

import { getI18n } from "./i18n";
import { createBrowserLocaleEnv, resolveLocale, writeLocalePreference } from "./localePreferences";
import type { Locale } from "./resources/registry";

interface I18nContextValue {
  locale: Locale;
  setLocale: (locale: Locale) => Promise<void>;
}

const I18nContext = createContext<I18nContextValue | null>(null);

/** Access the active locale and a `setLocale` that persists and applies it. */
export function useI18n(): I18nContextValue {
  const ctx = useContext(I18nContext);
  if (!ctx) {
    throw new Error("useI18n must be used within <I18nProvider>");
  }
  return ctx;
}

/**
 * Wraps the whole dashboard (root layout) as the outermost provider.
 *
 * First-screen strategy = language readiness gate (I18N_TECH_DESIGN §4 / ADR-03/04):
 *  - Resolve the target locale from the D5 preference chain.
 *  - Initialize the lazy i18next singleton and switch to the target language.
 *  - Only then render the business subtree, so no `t()`/`<Trans>` content is ever
 *    rendered before resources are ready (no raw key flash).
 *  - Sync `document.documentElement.lang` with the resolved language post-mount.
 * The English default is equally gated (sub-frame), since a non-initialised
 * i18next would otherwise expose raw keys for `en` too.
 */
export function I18nProvider({ children }: { children: ReactNode }) {
  const [i18n, setI18n] = useState<i18n | null>(null);
  const [locale, setLocaleState] = useState<Locale>("en");
  const [ready, setReady] = useState(false);

  useEffect(() => {
    let cancelled = false;
    const env = createBrowserLocaleEnv();
    const target = resolveLocale(env);

    void getI18n()
      .then(async (instance) => {
        if (cancelled) return;
        await instance.changeLanguage(target);
        if (cancelled) return;
        document.documentElement.lang = instance.language;
        setI18n(instance);
        setLocaleState(instance.language as Locale);
        setReady(true);
      })
      .catch(() => {
        // Pathological case: bundled static resources failed to initialise.
        // Do not render business content without an instance (that would expose
        // raw keys); surface the failure in the console.
        if (!cancelled) {
          // eslint-disable-next-line no-console
          console.error("[i18n] failed to initialise i18next instance");
        }
      });

    return () => {
      cancelled = true;
    };
  }, []);

  const setLocale = useCallback(
    async (next: Locale) => {
      const env = createBrowserLocaleEnv();
      writeLocalePreference(next, env);
      if (i18n) {
        await i18n.changeLanguage(next);
        document.documentElement.lang = i18n.language;
      }
      setLocaleState(next);
    },
    [i18n],
  );

  if (!ready || !i18n) {
    return null;
  }

  return (
    <I18nextProvider i18n={i18n}>
      <I18nContext.Provider value={{ locale, setLocale }}>{children}</I18nContext.Provider>
    </I18nextProvider>
  );
}
