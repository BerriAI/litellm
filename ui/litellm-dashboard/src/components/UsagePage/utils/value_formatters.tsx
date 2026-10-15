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

export function averageResponseTimeMs(totalResponseTimeMs: number, timedRequests: number): number | null {
  if (timedRequests <= 0) return null;
  return totalResponseTimeMs / timedRequests;
}

export function formatResponseTime(ms: number | null | undefined) {
  if (ms == null) return "-";
  if (ms < 1000) return `${Math.round(ms)}ms`;
  return `${(ms / 1000).toFixed(2)}s`;
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
