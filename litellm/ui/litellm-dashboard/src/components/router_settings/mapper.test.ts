import { describe, expect, it } from "vitest";
import { toRouterSettingsPayload } from "./mapper";

describe("router settings payload", () => {
  it("parses live values and omits settings owned by config or another tab", () => {
    const inputs: Record<string, string> = {
      allowed_fails: " 0 ",
      cooldown_time: " NULL ",
      model_group_alias: ' {"alias":"model"} ',
      custom_flag: " TrUe ",
      custom_name: "  model  ",
      lowest_latency_buffer: "0",
      ttl: "45",
    };

    expect(
      toRouterSettingsPayload(
        {
          routerSettings: {
            allowed_fails: 3,
            cooldown_time: 10,
            model_group_alias: {},
            custom_flag: false,
            custom_name: "",
            routing_strategy: "simple-shuffle",
            routing_strategy_args: { lowest_latency_buffer: 2, ttl: 30 },
            timeout: 45,
            retry_policy: {},
            routing_groups: [],
          },
          selectedStrategy: "latency-based-routing",
          enableTagFiltering: true,
        },
        { timeout: "config" },
        (name) => inputs[name],
      ),
    ).toEqual({
      router_settings: {
        allowed_fails: 0,
        cooldown_time: null,
        model_group_alias: { alias: "model" },
        custom_flag: true,
        custom_name: "model",
        routing_strategy: "latency-based-routing",
        routing_strategy_args: { lowest_latency_buffer: 0, ttl: 45 },
        enable_tag_filtering: true,
      },
    });
  });

  it("retains loaded values when live values are invalid or missing", () => {
    const inputs: Record<string, string> = { timeout: "invalid", model_group_alias: "{" };
    expect(
      toRouterSettingsPayload(
        {
          routerSettings: { timeout: 45, model_group_alias: { alias: "model" }, retry_after: 5 },
          selectedStrategy: null,
          enableTagFiltering: false,
        },
        {},
        (name) => inputs[name],
      ),
    ).toEqual({
      router_settings: {
        timeout: 45,
        model_group_alias: { alias: "model" },
        retry_after: 5,
        enable_tag_filtering: false,
      },
    });
  });

  it("uses zero for a blank numeric field, null for blank JSON, and omits blank latency arguments", () => {
    expect(
      toRouterSettingsPayload(
        {
          routerSettings: { timeout: 45, model_group_alias: {}, routing_strategy_args: { ttl: 30 } },
          selectedStrategy: "latency-based-routing",
          enableTagFiltering: false,
        },
        {},
        () => "",
      ),
    ).toEqual({
      router_settings: { timeout: 0, model_group_alias: null, routing_strategy_args: {}, enable_tag_filtering: false },
    });
  });
});
