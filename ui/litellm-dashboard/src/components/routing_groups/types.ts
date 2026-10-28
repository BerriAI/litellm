export type RoutingStrategy =
  | "simple-shuffle"
  | "least-busy"
  | "usage-based-routing"
  | "latency-based-routing"
  | "priority";

export interface RoutingGroup {
  group_name: string;
  models: string[];
  routing_strategy: RoutingStrategy | string;
  routing_strategy_args?: Record<string, unknown> | null;
  model_priorities?: Record<string, number> | null;
}
