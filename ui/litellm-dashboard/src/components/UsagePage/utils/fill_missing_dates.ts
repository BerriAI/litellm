import { formatDate } from "@/components/networking";
import { DailyData } from "../types";

const MS_PER_DAY = 86_400_000;

const startOfLocalDay = (date: Date): Date => new Date(date.getFullYear(), date.getMonth(), date.getDate());

const localDaysInRange = (from: Date, to: Date): string[] => {
  const start = startOfLocalDay(from);
  const dayCount = Math.round((startOfLocalDay(to).getTime() - start.getTime()) / MS_PER_DAY) + 1;
  return Array.from({ length: Math.max(dayCount, 0) }, (_, offset) =>
    formatDate(new Date(start.getFullYear(), start.getMonth(), start.getDate() + offset)),
  );
};

const zeroDailyData = (date: string): DailyData => ({
  date,
  metrics: {
    spend: 0,
    prompt_tokens: 0,
    completion_tokens: 0,
    total_tokens: 0,
    api_requests: 0,
    successful_requests: 0,
    failed_requests: 0,
    cache_read_input_tokens: 0,
    cache_creation_input_tokens: 0,
  },
  breakdown: { models: {}, model_groups: {}, mcp_servers: {}, providers: {}, api_keys: {}, entities: {} },
});

export const fillMissingDates = (data: DailyData[], from: Date | undefined, to: Date | undefined): DailyData[] => {
  if (!from || !to) return data;
  const rowsByDate = new Map(data.map((row) => [row.date, row]));
  return localDaysInRange(from, to).map((date) => rowsByDate.get(date) ?? zeroDailyData(date));
};
