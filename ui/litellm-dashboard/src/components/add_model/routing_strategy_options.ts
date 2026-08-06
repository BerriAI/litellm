export const ROUTING_STRATEGY_OPTIONS = [
  { value: "simple-shuffle", label: "Simple Shuffle (weighted random)" },
  { value: "latency-based-routing", label: "Latency-Based (fastest deployment)" },
  { value: "cost-based-routing", label: "Cost-Based (cheapest deployment)" },
  { value: "usage-based-routing-v2", label: "Usage-Based v2 (lowest TPM load)" },
  { value: "usage-based-routing", label: "Usage-Based v1 (lowest TPM load)" },
  { value: "least-busy", label: "Least-Busy (fewest in-flight requests)" },
] as const;

export const routingStrategyLabel = (value: string | undefined | null): string => {
  const match = ROUTING_STRATEGY_OPTIONS.find((o) => o.value === value);
  return match ? match.label : "Inherit router default";
};
