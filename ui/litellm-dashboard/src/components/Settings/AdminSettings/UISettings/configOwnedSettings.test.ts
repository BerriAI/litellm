import { describe, expect, it } from "vitest";

import { parseConfigOwnedUISettings } from "./configOwnedSettings";

describe("parseConfigOwnedUISettings", () => {
  it("returns only the keys whose source is config", () => {
    const owned = parseConfigOwnedUISettings({
      values: {},
      source: {
        team_admin_editable_team_fields: "config",
        disable_custom_api_keys: "config",
        enable_chat_ui: "db",
        scope_user_search_to_org: "default",
        moyai_url: "unset",
        future_setting: "some_new_source",
      },
    });

    expect([...owned].sort()).toEqual(["disable_custom_api_keys", "team_admin_editable_team_fields"]);
  });

  it.each([undefined, null, {}, { source: null }, { source: ["config"] }])(
    "treats a response without a usable source map (%j) as owning nothing",
    (response) => {
      expect(parseConfigOwnedUISettings(response).size).toBe(0);
    },
  );
});
