import type { TFunction } from "i18next";
import moment from "moment";

import i18n from "@/i18n/i18n";
import { formatDateWithEnglishPattern } from "@/i18n/formats";

// Add this function to format the time range display
export const getTimeRangeDisplay = (
  isCustomDate: boolean,
  startTime: string,
  endTime: string,
  { t = i18n.t, language = i18n.language }: { t?: TFunction; language?: string } = {},
) => {
  if (isCustomDate) {
    const format = (value: string) => {
      const date = moment(value);
      return formatDateWithEnglishPattern(date.toDate(), language, date.format("MMM D, h:mm A"), {
        month: "short",
        day: "numeric",
        hour: "numeric",
        minute: "2-digit",
      });
    };
    return `${format(startTime)} - ${format(endTime)}`;
  }

  const now = moment();
  const start = moment(startTime);
  const diffMinutes = now.diff(start, "minutes");

  // Use exact ranges to prevent drift
  if (diffMinutes >= 0 && diffMinutes < 2) return t("logs.request.timeRange.lastMinute");
  if (diffMinutes >= 2 && diffMinutes < 16) return t("logs.request.timeRange.last15Minutes");
  if (diffMinutes >= 16 && diffMinutes < 61) return t("logs.request.timeRange.lastHour");

  const diffHours = now.diff(start, "hours");
  if (diffHours >= 1 && diffHours < 5) return t("logs.request.timeRange.last4Hours");
  if (diffHours >= 5 && diffHours < 25) return t("logs.request.timeRange.last24Hours");
  if (diffHours >= 25 && diffHours < 169) return t("logs.request.timeRange.last7Days");
  const monthDay = (date: moment.Moment) =>
    date.toDate().toLocaleDateString(language, { month: "short", day: "numeric" });
  return `${monthDay(start)} - ${monthDay(now)}`;
};
