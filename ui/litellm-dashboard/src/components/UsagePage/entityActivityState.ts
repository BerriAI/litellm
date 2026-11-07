import { createParser, parseAsString } from "nuqs";
import { relativeTimeOptions } from "@/components/shared/date_range_presets";

export const parseActivityDay = createParser({
  parse(value) {
    if (!/^\d{4}-\d{2}-\d{2}$/.test(value)) return null;
    const date = new Date(`${value}T00:00:00Z`);
    return Number.isNaN(date.getTime()) || date.toISOString().slice(0, 10) !== value ? null : value;
  },
  serialize: (value: string) => value,
});

export const entityActivityParsers = {
  activity_from: parseActivityDay,
  activity_to: parseActivityDay,
  alert: parseAsString,
};

export const modelActivityParser = parseAsString.withOptions({ history: "push" });

export const localActivityDay = (date: Date): string =>
  `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, "0")}-${String(date.getDate()).padStart(2, "0")}`;

export function defaultActivityDays() {
  const to = new Date();
  const from = new Date(to);
  from.setUTCDate(from.getUTCDate() - 29);
  return { from: from.toISOString().slice(0, 10), to: to.toISOString().slice(0, 10) };
}

export const activityPresets = relativeTimeOptions.map((option) => ({
  ...option,
  getValue: () => {
    const to = new Date();
    const from = new Date(to);
    if (option.settingValue === "last_7_days") from.setUTCDate(from.getUTCDate() - 6);
    if (option.settingValue === "last_30_days") from.setUTCDate(from.getUTCDate() - 29);
    if (option.settingValue === "month_to_date") from.setUTCDate(1);
    if (option.settingValue === "year_to_date") from.setUTCMonth(0, 1);
    return {
      from: new Date(`${from.toISOString().slice(0, 10)}T00:00:00`),
      to: new Date(`${to.toISOString().slice(0, 10)}T23:59:59`),
    };
  },
}));
