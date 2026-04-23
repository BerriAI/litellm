import dayjs from "dayjs";

export function formatBudgetReset(iso: string | null | undefined): string | null {
  if (!iso) return null;
  const resetDate = dayjs(iso);
  if (!resetDate.isValid()) return null;

  const days = resetDate.diff(dayjs(), "day");
  if (days < 0) return `on ${resetDate.format("MMM D, YYYY")}`;
  if (days === 0) return "today";
  if (days < 7) return `in ${days} day${days === 1 ? "" : "s"}`;
  return `on ${resetDate.format("MMM D, YYYY")}`;
}
