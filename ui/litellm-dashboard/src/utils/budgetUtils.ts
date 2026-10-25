import dayjs from "dayjs";
import utc from "dayjs/plugin/utc";

dayjs.extend(utc);

export function formatBudgetReset(iso: string | null | undefined): string | null {
  if (!iso) return null;
  const resetDate = dayjs(iso);
  if (!resetDate.isValid()) return null;
  return resetDate.format("MMM D, YYYY");
}

export function formatBudgetResetUtc(iso: string | null | undefined): string | null {
  if (!iso) return null;
  const resetDate = dayjs.utc(iso);
  if (!resetDate.isValid()) return null;
  return resetDate.format("MMM D, YYYY");
}
