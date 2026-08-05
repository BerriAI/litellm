const STRATEGY_LABELS: Readonly<Record<string, string>> = {
  "simple-shuffle": "Simple Shuffle",
  "least-busy": "Least Busy",
  "usage-based-routing": "Usage Based",
  "latency-based-routing": "Latency Based",
};

export const formatStrategyLabel = (strategy: string): string => STRATEGY_LABELS[strategy] ?? strategy;
