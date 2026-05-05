export type RoutingStrategy =
  | "simple-shuffle"
  | "least-busy"
  | "usage-based-routing"
  | "latency-based-routing";

export interface RoutingGroup {
  group_name: string;
  models: string[];
  routing_strategy: RoutingStrategy | string;
  routing_strategy_args?: Record<string, unknown> | null;
}
