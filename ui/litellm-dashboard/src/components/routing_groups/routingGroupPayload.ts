import type { RoutingGroup } from "./types";

export const STRATEGIES_WITH_ARGS = new Set<string>(["latency-based-routing", "usage-based-routing"]);

export const GROUP_NAME_MAX_LENGTH = 64;

export interface ModelPriorityDraft {
  model: string;
  priority: string;
}

export interface RoutingGroupFormValues {
  group_name: string;
  models: string[];
  routing_strategy: string;
  routing_strategy_args: string;
  model_priorities: ModelPriorityDraft[];
}

export type RoutingGroupPayload =
  | { readonly ok: true; readonly group: RoutingGroup }
  | { readonly ok: false; readonly field: "routing_strategy_args" | "model_priorities"; readonly message: string };

export const prioritiesForModels = (models: string[], current: ModelPriorityDraft[]): ModelPriorityDraft[] =>
  models.map((model, index) => ({
    model,
    priority: current.find((entry) => entry.model === model)?.priority ?? String(index + 1),
  }));

export const toRoutingGroupFormValues = (
  group: RoutingGroup | null,
  availableStrategies: string[],
): RoutingGroupFormValues => ({
  group_name: group?.group_name ?? "",
  models: group?.models ?? [],
  routing_strategy: group?.routing_strategy ?? availableStrategies[0] ?? "simple-shuffle",
  routing_strategy_args: group?.routing_strategy_args ? JSON.stringify(group.routing_strategy_args, null, 2) : "",
  model_priorities:
    group?.routing_strategy === "priority"
      ? [
          ...group.models.map((model) => ({
            model,
            priority: Object.hasOwn(group.model_priorities ?? {}, model) ? String(group.model_priorities?.[model]) : "",
          })),
          ...Object.entries(group.model_priorities ?? {})
            .filter(([model]) => !group.models.includes(model))
            .map(([model, priority]) => ({ model, priority: String(priority) })),
        ]
      : prioritiesForModels(group?.models ?? [], []),
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

  if (values.routing_strategy === "priority") {
    const priorities = values.model_priorities;
    const configuredModels = new Set(priorities.map(({ model }) => model));
    const exactMembership =
      configuredModels.size === values.models.length && values.models.every((model) => configuredModels.has(model));
    if (priorities.length === 0 || priorities.length !== values.models.length || !exactMembership) {
      return {
        ok: false,
        field: "model_priorities",
        message: "Set a priority for each selected model and remove unused priorities",
      };
    }
    if (
      priorities.some(
        ({ priority }) =>
          !/^\d+$/.test(priority.trim()) || !Number.isSafeInteger(Number(priority)) || Number(priority) < 1,
      )
    ) {
      return {
        ok: false,
        field: "model_priorities",
        message: "Priorities must be whole numbers from 1 to 9007199254740991",
      };
    }
    return {
      ok: true,
      group: {
        ...base,
        routing_strategy_args: null,
        model_priorities: Object.fromEntries(priorities.map(({ model, priority }) => [model, Number(priority)])),
      },
    };
  }

  if (!args.trim()) {
    return { ok: true, group: { ...base, routing_strategy_args: null } };
  }

  try {
    return { ok: true, group: { ...base, routing_strategy_args: JSON.parse(args) as Record<string, unknown> } };
  } catch {
    return { ok: false, field: "routing_strategy_args", message: "Must be valid JSON" };
  }
};
