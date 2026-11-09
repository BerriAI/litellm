import type { ParseKeys, TFunction } from "i18next";

const STRATEGY_LABELS: Readonly<Record<string, ParseKeys>> = {
  "simple-shuffle": "routingGroups.strategies.simpleShuffle",
  "least-busy": "routingGroups.strategies.leastBusy",
  "usage-based-routing": "routingGroups.strategies.usageBased",
  "latency-based-routing": "routingGroups.strategies.latencyBased",
  priority: "routingGroups.strategies.priority",
};

export const formatStrategyLabel = (strategy: string, t: TFunction): string =>
  Object.hasOwn(STRATEGY_LABELS, strategy) ? t(STRATEGY_LABELS[strategy]) : strategy;
