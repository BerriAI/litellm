import { createParser, parseAsString } from "nuqs";

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
