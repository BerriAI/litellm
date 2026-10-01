import { createInstance, type i18n } from "i18next";
import { initReactI18next } from "react-i18next";

import { RESOURCES, supportedLngs } from "./resources/registry";

/**
 * Lazy i18next singleton. Initialization happens once and is reused across
 * HMR / re-mounts; callers `await getI18n()` rather than importing a module-top
 * instance (I18N_TECH_DESIGN §2.2) so the static-export build never initializes
 * i18next during compilation.
 *
 * Note (i18next v26): `.init()`'s promise resolves to the bound `t` function,
 * not the instance, so we await it only to observe readiness and resolve the
 * cached instance itself.
 */
let pending: Promise<i18n> | null = null;

export function getI18n(): Promise<i18n> {
  if (!pending) {
    const instance = createInstance();

    const initPromise = instance.use(initReactI18next).init({
      resources: RESOURCES,
      supportedLngs: [...supportedLngs],
      fallbackLng: "en",
      load: "currentOnly",
      nonExplicitSupportedLngs: false,
      defaultNS: "common",
      ns: ["common"],
      // Never render a raw English sentence as a "missing-key fallback"; a
      // fully-missing key renders nothing (ADR-03 §5.1) and warns in dev so it
      // surfaces early. English is the type source, so this is a last resort.
      returnNull: true,
      returnEmptyString: false,
      interpolation: { escapeValue: false },
      react: { useSuspense: false },
      missingKeyHandler: (_lngs, _ns, key) => {
        if (process.env.NODE_ENV === "development") {
          // eslint-disable-next-line no-console
          console.warn(`[i18n] missing key: ${key}`);
        }
      },
    });

    pending = initPromise.then(() => instance);
  }
  return pending;
}
