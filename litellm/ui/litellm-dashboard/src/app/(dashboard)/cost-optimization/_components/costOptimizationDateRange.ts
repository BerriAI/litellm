import { endOfDay, startOfDay, subDays } from "date-fns";

export const costOptimizationDateRange = (now: Date = new Date()): { from: Date; to: Date } => ({
  from: startOfDay(subDays(now, 6)),
  to: endOfDay(now),
});
