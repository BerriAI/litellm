import { BreakdownMetrics, DailyData, KeyMetricWithMetadata, TagUsage } from "@/components/UsagePage/types";

export type ExtendedDailyData = DailyData & {
  breakdown: BreakdownMetrics;
};

export type ModelBreakdownKey = "models" | "model_groups";

export interface ProviderSpendRow extends Record<string, unknown> {
  provider: string;
  spend: number;
  requests: number;
  successful_requests: number;
  failed_requests: number;
  tokens: number;
}

export const getTopModels = (
  results: ExtendedDailyData[],
  modelBreakdownKey: ModelBreakdownKey,
  topModelsLimit: number,
) => {
  const modelSpend: { [key: string]: any } = {};
  results.forEach((day) => {
    Object.entries(day.breakdown[modelBreakdownKey] || {}).forEach(([model, metrics]) => {
      if (!modelSpend[model]) {
        modelSpend[model] = {
          spend: 0,
          requests: 0,
          successful_requests: 0,
          failed_requests: 0,
          tokens: 0,
        };
      }
      try {
        modelSpend[model].spend += metrics.metrics.spend;
      } catch (e) {
        console.error(`Error adding spend for ${model}: ${e}, got metrics: ${JSON.stringify(metrics)}`);
      }
      modelSpend[model].requests += metrics.metrics.api_requests;
      modelSpend[model].successful_requests += metrics.metrics.successful_requests;
      modelSpend[model].failed_requests += metrics.metrics.failed_requests;
      modelSpend[model].tokens += metrics.metrics.total_tokens;
    });
  });

  return Object.entries(modelSpend)
    .map(([model, metrics]) => ({
      key: model,
      ...metrics,
    }))
    .sort((a, b) => b.spend - a.spend)
    .slice(0, topModelsLimit);
};

export const getTopAgents = (results: ExtendedDailyData[], topAgentsLimit: number) => {
  const agentSpend: { [key: string]: any } = {};
  results.forEach((day) => {
    Object.entries(day.breakdown.entities || {}).forEach(([agentId, data]) => {
      if (!agentSpend[agentId]) {
        agentSpend[agentId] = {
          spend: 0,
          requests: 0,
          successful_requests: 0,
          failed_requests: 0,
          tokens: 0,
          agent_name: (data.metadata as any)?.agent_name || agentId,
        };
      }
      agentSpend[agentId].spend += data.metrics.spend;
      agentSpend[agentId].requests += data.metrics.api_requests;
      agentSpend[agentId].successful_requests += data.metrics.successful_requests;
      agentSpend[agentId].failed_requests += data.metrics.failed_requests;
      agentSpend[agentId].tokens += data.metrics.total_tokens;
    });
  });

  return Object.entries(agentSpend)
    .map(([agentId, metrics]) => ({
      key: metrics.agent_name,
      ...metrics,
    }))
    .sort((a, b) => b.spend - a.spend)
    .slice(0, topAgentsLimit);
};

export const getTopAPIKeys = (results: ExtendedDailyData[], topKeysLimit: number) => {
  const keySpend: { [key: string]: KeyMetricWithMetadata } = {};
  results.forEach((day) => {
    const { breakdown } = day;
    const { entities } = breakdown;
    const tagDictionary = Object.keys(entities).reduce((acc: { [key: string]: TagUsage[] }, entity) => {
      const { api_key_breakdown } = entities[entity];
      Object.keys(api_key_breakdown).forEach((key) => {
        const tagUsage = { tag: entity, usage: api_key_breakdown[key].metrics.spend };
        if (acc[key]) {
          acc[key].push(tagUsage);
        } else {
          acc[key] = [tagUsage];
        }
      });
      return acc;
    }, {});
    Object.entries(day.breakdown.api_keys || {}).forEach(([key, metrics]) => {
      if (!keySpend[key]) {
        keySpend[key] = {
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
          metadata: {
            key_alias: metrics.metadata.key_alias,
            team_id: metrics.metadata.team_id || null,
            tags: tagDictionary[key] || [],
          },
        };
      }
      keySpend[key].metrics.spend += metrics.metrics.spend;
      keySpend[key].metrics.prompt_tokens += metrics.metrics.prompt_tokens;
      keySpend[key].metrics.completion_tokens += metrics.metrics.completion_tokens;
      keySpend[key].metrics.total_tokens += metrics.metrics.total_tokens;
      keySpend[key].metrics.api_requests += metrics.metrics.api_requests;
      keySpend[key].metrics.successful_requests += metrics.metrics.successful_requests;
      keySpend[key].metrics.failed_requests += metrics.metrics.failed_requests;
      keySpend[key].metrics.cache_read_input_tokens += metrics.metrics.cache_read_input_tokens || 0;
      keySpend[key].metrics.cache_creation_input_tokens += metrics.metrics.cache_creation_input_tokens || 0;
    });
  });

  return Object.entries(keySpend)
    .map(([api_key, metrics]) => ({
      api_key,
      key_alias: metrics.metadata.key_alias || "-", // Using truncated key as alias
      tags: metrics.metadata.tags || "-",
      spend: metrics.metrics.spend,
    }))
    .sort((a, b) => b.spend - a.spend)
    .slice(0, topKeysLimit);
};

export const getProviderSpend = (results: ExtendedDailyData[]): ProviderSpendRow[] => {
  const providerSpend: Record<string, ProviderSpendRow> = {};
  results.forEach((day) => {
    Object.entries(day.breakdown.providers || {}).forEach(([provider, metrics]) => {
      if (!providerSpend[provider]) {
        providerSpend[provider] = {
          provider,
          spend: 0,
          requests: 0,
          successful_requests: 0,
          failed_requests: 0,
          tokens: 0,
        };
      }
      try {
        providerSpend[provider].spend += metrics.metrics.spend;
        providerSpend[provider].requests += metrics.metrics.api_requests;
        providerSpend[provider].successful_requests += metrics.metrics.successful_requests;
        providerSpend[provider].failed_requests += metrics.metrics.failed_requests;
        providerSpend[provider].tokens += metrics.metrics.total_tokens;
      } catch (e) {
        console.error(`Error processing provider ${provider}: ${e}`);
      }
    });
  });

  return Object.values(providerSpend)
    .filter((provider) => provider.spend > 0)
    .sort((a, b) => b.spend - a.spend);
};
