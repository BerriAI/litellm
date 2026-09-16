import { describe, expect, it } from "vitest";
import { RESOURCES, NAMESPACES } from "./resources/registry";

// Resource-shape tests. With i18next v26's typed-qualified-key support being
// fragile (and breaking `next build`), `types.d.ts` types keys loosely and key
// correctness is delegated to runtime readiness + A7's `check-keys` CI gate.
// These tests keep the registry honest at runtime: every locale exposes every
// registered namespace, and the seed keys the platform actually uses exist in
// both locales — so a removed/moved resource fails here, not silently at runtime.
describe("i18n resource registry", () => {
  it("exposes every registered namespace for every locale", () => {
    for (const locale of Object.keys(RESOURCES) as (keyof typeof RESOURCES)[]) {
      for (const name of NAMESPACES) {
        expect(RESOURCES[locale], `${locale}.${name}`).toHaveProperty(name);
      }
    }
  });

  it("has the common seed keys used by the platform in en and zh-CN", () => {
    for (const locale of ["en", "zh-CN"] as const) {
      const common = RESOURCES[locale].common as Record<string, unknown>;
      expect(common).toHaveProperty("language.name");
      expect(common).toHaveProperty("languages.en");
      expect(common).toHaveProperty("languages.zh-CN");
    }
  });

  it("has the navigation seed key in en and zh-CN", () => {
    for (const locale of ["en", "zh-CN"] as const) {
      const nav = RESOURCES[locale].navigation as Record<string, unknown>;
      expect(nav).toHaveProperty("dashboard");
    }
  });
});
