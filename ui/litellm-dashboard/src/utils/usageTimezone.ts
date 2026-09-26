export const usageCalendarDate = (instant: Date, reportingTimezone?: string): Date => {
  if (!reportingTimezone) return instant;
  const options: Intl.DateTimeFormatOptions = {
    timeZone: reportingTimezone,
    year: "numeric",
    month: "numeric",
    day: "numeric",
  };
  const parts = Object.fromEntries(
    new Intl.DateTimeFormat("en-US", options).formatToParts(instant).map(({ type, value }) => [type, value]),
  );
  return new Date(Number(parts.year), Number(parts.month) - 1, Number(parts.day));
};

export const usageTimezoneLabel = (reportingTimezone?: string): string => reportingTimezone ?? "UTC";
