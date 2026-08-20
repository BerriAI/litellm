import type { RoutingGroup } from "./types";

export const STRATEGIES_WITH_ARGS = new Set<string>(["latency-based-routing", "usage-based-routing"]);

export const GROUP_NAME_PATTERN = /^[A-Za-z0-9._-]+$/;
export const GROUP_NAME_MAX_LENGTH = 64;

export interface RoutingGroupFormValues {
  group_name: string;
  models: string[];
  routing_strategy: string;
  routing_strategy_args: string;
}

export type RoutingGroupPayload =
  | { readonly ok: true; readonly group: RoutingGroup }
  | { readonly ok: false; readonly argsError: string };

export const toRoutingGroupFormValues = (
  group: RoutingGroup | null,
  availableStrategies: string[],
): RoutingGroupFormValues => ({
  group_name: group?.group_name ?? "",
  models: group?.models ?? [],
  routing_strategy: group?.routing_strategy ?? availableStrategies[0] ?? "simple-shuffle",
  routing_strategy_args: group?.routing_strategy_args ? JSON.stringify(group.routing_strategy_args, null, 2) : "",
});

export const argsForStrategy = (routingStrategy: string, routingStrategyArgs: string): string =>
  STRATEGIES_WITH_ARGS.has(routingStrategy) ? routingStrategyArgs : "";

export const buildRoutingGroupPayload = (values: RoutingGroupFormValues): RoutingGroupPayload => {
  const base = {
    group_name: values.group_name.trim(),
    models: values.models,
    routing_strategy: values.routing_strategy,
  };
  const args = argsForStrategy(values.routing_strategy, values.routing_strategy_args);

  if (!args.trim()) {
    return { ok: true, group: { ...base, routing_strategy_args: null } };
  }

  try {
    return { ok: true, group: { ...base, routing_strategy_args: JSON.parse(args) as Record<string, unknown> } };
  } catch {
    return { ok: false, argsError: "Must be valid JSON" };
  }
};
