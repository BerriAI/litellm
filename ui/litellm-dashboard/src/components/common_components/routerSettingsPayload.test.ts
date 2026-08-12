import { describe, expect, it } from "vitest";
import { hasRouterSettings, routerSettingsUpdate } from "./routerSettingsPayload";

describe("hasRouterSettings", () => {
  it("should treat unset, empty and all-null settings as absent", () => {
    expect(hasRouterSettings(undefined)).toBe(false);
    expect(hasRouterSettings(null)).toBe(false);
    expect(hasRouterSettings({})).toBe(false);
    expect(
      hasRouterSettings({ num_retries: null, fallbacks: [], model_group_alias: {}, enable_tag_filtering: false }),
    ).toBe(false);
  });

  it("should detect configured settings", () => {
    expect(hasRouterSettings({ num_retries: 3 })).toBe(true);
    expect(hasRouterSettings({ fallbacks: [{ "gpt-4": ["gpt-4o"] }] })).toBe(true);
    expect(hasRouterSettings({ enable_tag_filtering: true })).toBe(true);
    expect(hasRouterSettings({ num_retries: 0 })).toBe(true);
  });
});

describe("routerSettingsUpdate", () => {
  const fallbacks = [{ "gpt-4": ["gpt-4o"] }];

  it("should send the edited settings when the user configured something", () => {
    expect(routerSettingsUpdate({ fallbacks }, null)).toEqual({ fallbacks });
  });

  it("should send the edited settings unchanged so an unrelated edit keeps stored fallbacks", () => {
    expect(routerSettingsUpdate({ fallbacks, num_retries: 2 }, { fallbacks, num_retries: 2 })).toEqual({
      fallbacks,
      num_retries: 2,
    });
  });

  it("should send cleared settings so removing every fallback reaches the server", () => {
    expect(routerSettingsUpdate({ fallbacks: null, num_retries: null }, { fallbacks })).toEqual({
      fallbacks: null,
      num_retries: null,
    });
  });

  it("should leave the field off when nothing is stored and nothing was configured", () => {
    expect(routerSettingsUpdate({ fallbacks: null, num_retries: null }, {})).toBeUndefined();
    expect(routerSettingsUpdate(undefined, { fallbacks })).toBeUndefined();
  });
});
