import moment from "moment";

const parseStoredTimestamp = (value: string): Date => {
  const normalized = value.includes("T") ? value : value.replace(" ", "T").slice(0, 23) + "Z";
  return new Date(normalized);
};

export function formatActivityTimestamp(value: string): string {
  const date = parseStoredTimestamp(value);
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

export const RUN_TIMESTAMP_FORMAT = "MMM DD HH:mm:ss.SSS";

export function formatRunTimestamp(value: string): string {
  const date = parseStoredTimestamp(value);
  if (Number.isNaN(date.getTime())) return value;
  return moment(date).format(RUN_TIMESTAMP_FORMAT);
}

export function localTimeZoneAbbreviation(at: Date = new Date()): string {
  const part = new Intl.DateTimeFormat(undefined, { timeZoneName: "short" })
    .formatToParts(at)
    .find((p) => p.type === "timeZoneName");
  return part?.value ?? "";
}
