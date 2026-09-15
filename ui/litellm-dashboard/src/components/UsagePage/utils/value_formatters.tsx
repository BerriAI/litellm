export function valueFormatter(number: number) {
  if (number >= 1_000_000_000) {
    return (number / 1_000_000_000).toFixed(2) + "B";
  }
  if (number >= 1_000_000) {
    return (number / 1_000_000).toFixed(2) + "M";
  }
  if (number >= 1000) {
    return number / 1000 + "k";
  }
  return number.toString();
}

export function formatAvgLatency(totalLatencyMs: number | undefined, successfulRequests: number): string {
  if (!totalLatencyMs || totalLatencyMs <= 0 || successfulRequests <= 0) return "-";
  const avgMs = totalLatencyMs / successfulRequests;
  if (avgMs >= 1000) return `${(avgMs / 1000).toFixed(2)} s`;
  return `${Math.round(avgMs)} ms`;
}

export function valueFormatterSpend(number: number) {
  if (number === 0) return "$0";
  if (number >= 1_000_000_000) {
    return "$" + parseFloat((number / 1_000_000_000).toFixed(2)) + "B";
  }
  if (number >= 1_000_000) {
    return "$" + parseFloat((number / 1_000_000).toFixed(2)) + "M";
  }
  if (number >= 1000) {
    return "$" + number / 1000 + "k";
  }
  return "$" + number;
}
