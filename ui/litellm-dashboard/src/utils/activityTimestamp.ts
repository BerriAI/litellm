export function formatActivityTimestamp(value: string): string {
  const normalized = value.includes("T") ? value : value.replace(" ", "T").slice(0, 23) + "Z";
  const date = new Date(normalized);
  if (Number.isNaN(date.getTime())) return value;
  const options: Intl.DateTimeFormatOptions = {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
    second: "2-digit",
    timeZoneName: "short",
  };
  return date.toLocaleString(undefined, options);
}
