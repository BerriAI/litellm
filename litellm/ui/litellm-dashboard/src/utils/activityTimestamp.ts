import moment from "moment";
import { displayTimestamp } from "@/i18n/displayDate";
import { formatDateWithEnglishPattern } from "@/i18n/formats";
import i18n from "@/i18n/i18n";

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
  return displayTimestamp(date, i18n.language, options);
}

export const RUN_TIMESTAMP_FORMAT = "MMM DD HH:mm:ss.SSS";
const RUN_TIMESTAMP_OPTIONS = {
  month: "short",
  day: "numeric",
  hour: "2-digit",
  minute: "2-digit",
  second: "2-digit",
  fractionalSecondDigits: 3,
} as const;

export function formatRunTimestamp(value: string): string {
  const date = parseStoredTimestamp(value);
  if (Number.isNaN(date.getTime())) return value;
  return formatDateWithEnglishPattern(
    date,
    i18n.language,
    moment(date).format(RUN_TIMESTAMP_FORMAT),
    RUN_TIMESTAMP_OPTIONS,
  );
}

export function localTimeZoneAbbreviation(at: Date = new Date()): string {
  const part = new Intl.DateTimeFormat(undefined, { timeZoneName: "short" })
    .formatToParts(at)
    .find((p) => p.type === "timeZoneName");
  return part?.value ?? "";
}
