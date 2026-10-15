import { describe, expect, it } from "vitest";
import { hasRouterSettings, routerSettingsEditorValue, routerSettingsUpdate } from "./routerSettingsPayload";

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

describe("routerSettingsEditorValue", () => {
  it("should hand the editor only the fields it renders", () => {
    expect(
      routerSettingsEditorValue({
        num_retries: 2,
        tag_routing_prefix: "team-",
        fallbacks: [{ "gpt-4": ["gpt-4o"] }],
      }),
    ).toStrictEqual({ router_settings: { num_retries: 2, fallbacks: [{ "gpt-4": ["gpt-4o"] }] } });
  });

  it("should keep an explicitly stored null so the editor renders it as empty", () => {
    expect(routerSettingsEditorValue({ num_retries: null })).toStrictEqual({ router_settings: { num_retries: null } });
  });

  it("should leave the editor uninitialised when the key has no stored settings", () => {
    expect(routerSettingsEditorValue(null)).toBeUndefined();
    expect(routerSettingsEditorValue(undefined)).toBeUndefined();
  });
});

describe("routerSettingsUpdate", () => {
  const fallbacks = [{ "gpt-4": ["gpt-4o"] }];
  // Accepted by UpdateRouterConfig on /key/update but not rendered by the accordion.
  const unsupported = {
    tag_routing_prefix: "team-",
    model_group_retry_policy: { "gpt-4": { TimeoutErrorRetries: 2 } },
  };

  // The editor is a fixed-field form, so the merge has to hold two opposing guarantees at once:
  // routing fields it cannot render survive, and a field it does render that the admin emptied
  // is still sent as null. Orderings vary because an object spread resolves collisions by position.
  const storedOrderings: Array<[string, Record<string, unknown>]> = [
    ["unsupported fields first", { ...unsupported, num_retries: 2, fallbacks }],
    ["unsupported fields last", { num_retries: 2, fallbacks, ...unsupported }],
    [
      "unsupported fields interleaved",
      {
        tag_routing_prefix: unsupported.tag_routing_prefix,
        num_retries: 2,
        model_group_retry_policy: unsupported.model_group_retry_policy,
        fallbacks,
      },
    ],
  ];

  it.each(storedOrderings)(
    "should keep unsupported stored fields and still clear an emptied editor field (%s)",
    (_ordering, stored) => {
      const result = routerSettingsUpdate({ num_retries: 4, fallbacks: null }, stored);

      expect(result).toMatchObject({ ...unsupported, num_retries: 4, fallbacks: null });
    },
  );

  it("should send an empty object, not a null blob, when the last stored setting is cleared", () => {
    expect(routerSettingsUpdate({ fallbacks: null, num_retries: null }, { fallbacks, num_retries: 2 })).toEqual({});
  });

  it("should keep clearing owned fields explicitly while an unsupported setting still stands", () => {
    expect(routerSettingsUpdate({ fallbacks: null, num_retries: null }, { fallbacks, ...unsupported })).toMatchObject({
      ...unsupported,
      fallbacks: null,
      num_retries: null,
    });
  });

  it("should null every editor-owned field the editor left out", () => {
    expect(routerSettingsUpdate({ fallbacks: null }, { timeout: 30, ...unsupported })).toEqual({
      ...unsupported,
      routing_strategy: null,
      allowed_fails: null,
      cooldown_time: null,
      num_retries: null,
      timeout: null,
      retry_after: null,
      fallbacks: null,
      context_window_fallbacks: null,
      retry_policy: null,
      model_group_alias: null,
      enable_tag_filtering: null,
      routing_strategy_args: null,
    });
  });

  it("should send the edited settings when the user configured something", () => {
    expect(routerSettingsUpdate({ fallbacks }, null)).toMatchObject({ fallbacks });
  });

  it("should leave the field off when nothing is stored and nothing was configured", () => {
    expect(routerSettingsUpdate({ fallbacks: null, num_retries: null }, {})).toBeUndefined();
    expect(routerSettingsUpdate(undefined, { fallbacks })).toBeUndefined();
  });
});
