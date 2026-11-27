import { isOwnedByConfig, type RouterSettingsSource } from "./ConfigOwned";
import type { RouterSettingsFormValue } from "./RouterSettingsForm";

interface RoutingStrategyArgs {
  ttl?: number;
  lowest_latency_buffer?: number;
}

const numberKeys = new Set(["allowed_fails", "cooldown_time", "num_retries", "timeout", "retry_after"]);
const jsonKeys = new Set(["model_group_alias"]);
const tabOwnedKeys = new Set(["retry_policy", "model_group_retry_policy", "routing_groups"]);

const parseInputValue = (key: string, raw: string | undefined, fallback: unknown) => {
  if (raw === undefined) return fallback;

  const v = raw.trim();

  if (v.toLowerCase() === "null") return null;

  if (numberKeys.has(key)) {
    const n = Number(v);
    return Number.isNaN(n) ? fallback : n;
  }

  if (jsonKeys.has(key)) {
    if (v === "") return null;
    try {
      return JSON.parse(v);
    } catch {
      return fallback;
    }
  }

  if (v.toLowerCase() === "true") return true;
  if (v.toLowerCase() === "false") return false;

  return v;
};

export function toRouterSettingsPayload(
  formValue: RouterSettingsFormValue,
  routerSettingsSource: RouterSettingsSource,
  readInput: (name: string) => string | undefined,
) {
  const settingsToUpdate = {
    ...formValue.routerSettings,
    enable_tag_filtering: formValue.enableTagFiltering,
  };

  const updatedVariables = Object.fromEntries(
    Object.entries(settingsToUpdate)
      .map(([key, value]) => {
        if (tabOwnedKeys.has(key) || isOwnedByConfig(routerSettingsSource, key)) {
          return null;
        }
        if (key !== "routing_strategy_args" && key !== "routing_strategy" && key !== "enable_tag_filtering") {
          const parsed = parseInputValue(key, readInput(key), value);
          return [key, parsed];
        } else if (key === "routing_strategy") {
          return [key, formValue.selectedStrategy];
        } else if (key === "enable_tag_filtering") {
          return [key, formValue.enableTagFiltering];
        } else if (key === "routing_strategy_args" && formValue.selectedStrategy === "latency-based-routing") {
          const setRoutingStrategyArgs: RoutingStrategyArgs = {};
          const lowestLatencyBuffer = readInput("lowest_latency_buffer");
          const ttl = readInput("ttl");

          if (lowestLatencyBuffer) {
            setRoutingStrategyArgs["lowest_latency_buffer"] = Number(lowestLatencyBuffer);
          }

          if (ttl) {
            setRoutingStrategyArgs["ttl"] = Number(ttl);
          }

          return ["routing_strategy_args", setRoutingStrategyArgs];
        }
        return null;
      })
      .filter((entry) => entry !== null && entry !== undefined) as Iterable<[string, unknown]>,
  );

  return { router_settings: updatedVariables };
}
