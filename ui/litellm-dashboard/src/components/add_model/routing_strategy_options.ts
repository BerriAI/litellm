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

export const hasRoutingStrategyArgs = (args: object | undefined | null): boolean => Object.keys(args ?? {}).length > 0;

export const formItemValidateJSONObject = (_: unknown, value: string) => {
  if (!value) {
    return Promise.resolve();
  }
  try {
    const parsed = JSON.parse(value);
    if (parsed === null || Array.isArray(parsed) || typeof parsed !== "object") {
      return Promise.reject('Must be a JSON object, e.g. {"ttl": 3600}');
    }
    return Promise.resolve();
  } catch (error) {
    return Promise.reject("Please enter valid JSON");
  }
};
