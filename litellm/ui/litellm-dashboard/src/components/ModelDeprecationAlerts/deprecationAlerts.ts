import type { components } from "@/lib/http/schema";

type ModelDeprecation = components["schemas"]["ModelDeprecationInfo"];
type DailySpendData = components["schemas"]["DailySpendData"];

export interface DeprecationAlert {
  id: string;
  model: ModelDeprecation;
  usageModel: string;
  requests: number;
  lastUsedDate: string;
  lastDayRequests: number;
  history?: { date: string; value: number }[];
  state?: components["schemas"]["DeprecationAlertState"];
}

export function deprecationUsageRange(today: Date) {
  const end = new Date(today);
  const start = new Date(today);
  start.setUTCDate(start.getUTCDate() - 29);
  return { start_date: start.toISOString().slice(0, 10), end_date: end.toISOString().slice(0, 10) };
}

/** Match actual model usage, rather than model-group totals that may include healthy deployments. */
export function usedDeprecationAlerts(
  models: readonly ModelDeprecation[],
  days: readonly DailySpendData[],
): DeprecationAlert[] {
  const alerts: DeprecationAlert[] = [];
  const seen = new Set<string>();
  for (const model of models) {
    const baseModel = model.litellm_model ?? model.model_name;
    const candidates = new Set([baseModel]);
    if (model.litellm_provider && !baseModel.startsWith(`${model.litellm_provider}/`)) {
      candidates.add(`${model.litellm_provider}/${baseModel}`);
    }
    for (const usageModel of candidates) {
      const id = JSON.stringify([usageModel, model.deprecation_date, model.status]);
      if (seen.has(id)) continue;
      seen.add(id);
      let requests = 0;
      let lastUsedDate = "";
      let lastDayRequests = 0;
      for (const day of days) {
        const metrics = day.breakdown?.models?.[usageModel]?.metrics;
        if (!metrics || metrics.api_requests <= 0) continue;
        requests += metrics.api_requests;
        if (day.date > lastUsedDate) {
          lastUsedDate = day.date;
          lastDayRequests = metrics.api_requests;
        } else if (day.date === lastUsedDate) {
          lastDayRequests += metrics.api_requests;
        }
      }
      if (requests > 0) {
        const history = Array.from(new Set(days.map((day) => day.date)))
          .sort()
          .flatMap((date) => {
            const recorded = days.filter((day) => day.date === date && day.breakdown?.models?.[usageModel]);
            return recorded.length
              ? [
                  {
                    date,
                    value: recorded.reduce(
                      (sum, day) => sum + (day.breakdown?.models?.[usageModel]?.metrics.api_requests ?? 0),
                      0,
                    ),
                  },
                ]
              : [];
          });
        const alert = { id, model, usageModel, requests, lastUsedDate, lastDayRequests, history };
        alerts.push(alert);
      }
    }
  }
  return alerts.sort((a, b) => a.model.deprecation_date.localeCompare(b.model.deprecation_date));
}
