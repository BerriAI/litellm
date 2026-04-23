import { formatNumberWithCommas } from "@/utils/dataUtils";
import type { DateRangePickerValue } from "@tremor/react";
import Papa from "papaparse";
import type { EntityBreakdown, EntitySpendData, EntityType, ExportMetadata, ExportScope } from "./types";

// Resolve display name for an entity. For teams the teamAliasMap provides
// a human-readable alias; for every other entity type the entity key itself
// (tag name, org id, customer id, …) is already the correct label.
const resolveEntityDisplay = (
  entity: string,
  teamAliasMap: Record<string, string>,
): { id: string; alias: string } => ({
  id: entity,
  alias: teamAliasMap[entity] || entity,
});

// Mirrors backend SpendMetrics fields (litellm/types/activity_tracking.py).
// If the backend adds a field, add it here too.
const METRIC_KEYS = [
  "spend", "api_requests", "successful_requests", "failed_requests",
  "total_tokens", "prompt_tokens", "completion_tokens",
  "cache_read_input_tokens", "cache_creation_input_tokens",
] as const;

// When breakdown.entities is empty (aggregated endpoint), reconstruct entities
// from breakdown.api_keys by grouping on metadata.team_id.
const aggregateApiKeysIntoEntities = (breakdown: Record<string, any>): Record<string, any> => {
  const apiKeys = breakdown.api_keys;
  if (!apiKeys || Object.keys(apiKeys).length === 0) return {};

  const grouped: Record<string, any> = {};

  for (const [keyId, keyData] of Object.entries<any>(apiKeys)) {
    const teamId = keyData?.metadata?.team_id || "Unassigned";
    if (!grouped[teamId]) {
      grouped[teamId] = {
        metrics: Object.fromEntries(METRIC_KEYS.map((k) => [k, 0])),
        api_key_breakdown: {},
      };
    }
    const m = grouped[teamId].metrics;
    const km = keyData?.metrics || {};
    for (const k of METRIC_KEYS) {
      m[k] += km[k] || 0;
    }
    grouped[teamId].api_key_breakdown[keyId] = keyData;
  }

  return grouped;
};

// Returns breakdown.entities if populated, otherwise falls back to
// reconstructing entities from breakdown.api_keys.
export const resolveEntities = (breakdown: Record<string, any>): Record<string, any> => {
  const entities = breakdown.entities;
  if (entities && Object.keys(entities).length > 0) return entities;
  return aggregateApiKeysIntoEntities(breakdown);
};

export const getEntityBreakdown = (
  spendData: EntitySpendData,
  teamAliasMap: Record<string, string> = {},
): EntityBreakdown[] => {
  const entitySpend: { [key: string]: EntityBreakdown } = {};

  spendData.results.forEach((day) => {
    Object.entries(resolveEntities(day.breakdown)).forEach(([entity, data]: [string, any]) => {
      const { id, alias } = resolveEntityDisplay(entity, teamAliasMap);

      if (!entitySpend[entity]) {
        entitySpend[entity] = {
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
            alias,
            id,
          },
        };
      }
      entitySpend[entity].metrics.spend += data.metrics.spend;
      entitySpend[entity].metrics.api_requests += data.metrics.api_requests;
      entitySpend[entity].metrics.successful_requests += data.metrics.successful_requests;
      entitySpend[entity].metrics.failed_requests += data.metrics.failed_requests;
      entitySpend[entity].metrics.total_tokens += data.metrics.total_tokens;
      entitySpend[entity].metrics.prompt_tokens += data.metrics.prompt_tokens || 0;
      entitySpend[entity].metrics.completion_tokens += data.metrics.completion_tokens || 0;
      entitySpend[entity].metrics.cache_read_input_tokens += data.metrics.cache_read_input_tokens || 0;
      entitySpend[entity].metrics.cache_creation_input_tokens += data.metrics.cache_creation_input_tokens || 0;
    });
  });

  return Object.values(entitySpend).sort((a, b) => b.metrics.spend - a.metrics.spend);
};

export const generateDailyData = (
  spendData: EntitySpendData,
  entityLabel: string,
  teamAliasMap: Record<string, string> = {},
): any[] => {
  const dailyBreakdown: any[] = [];

  spendData.results.forEach((day) => {
    Object.entries(resolveEntities(day.breakdown)).forEach(([entity, data]: [string, any]) => {
      const { id, alias } = resolveEntityDisplay(entity, teamAliasMap);

      dailyBreakdown.push({
        Date: day.date,
        [entityLabel]: alias,
        [`${entityLabel} ID`]: id,
        "Spend ($)": formatNumberWithCommas(data.metrics.spend, 4),
        Requests: data.metrics.api_requests,
        "Successful Requests": data.metrics.successful_requests,
        "Failed Requests": data.metrics.failed_requests,
        "Total Tokens": data.metrics.total_tokens,
        "Prompt Tokens": data.metrics.prompt_tokens || 0,
        "Completion Tokens": data.metrics.completion_tokens || 0,
      });
    });
  });

  return dailyBreakdown.sort((a, b) => new Date(a.Date).getTime() - new Date(b.Date).getTime());
};

export const generateDailyWithKeysData = (
  spendData: EntitySpendData,
  entityLabel: string,
  teamAliasMap: Record<string, string> = {},
): any[] => {
  // Aggregate by unique (Date, Entity ID, Key ID) combination to prevent duplicates
  const aggregatedData: {
    [key: string]: {
      Date: string;
      entityId: string;
      entityAlias: string;
      keyId: string;
      keyAlias: string | null;
      metrics: {
        spend: number;
        api_requests: number;
        successful_requests: number;
        failed_requests: number;
        total_tokens: number;
        prompt_tokens: number;
        completion_tokens: number;
      };
    };
  } = {};

  spendData.results.forEach((day) => {
    Object.entries(resolveEntities(day.breakdown)).forEach(([entity, data]: [string, any]) => {
      const { id: entityId, alias: entityAlias } = resolveEntityDisplay(entity, teamAliasMap);
      const apiKeyBreakdown = data.api_key_breakdown || {};

      // Iterate through each API key in the breakdown
      Object.entries(apiKeyBreakdown).forEach(([keyId, keyData]: [string, any]) => {
        const keyAlias = keyData?.metadata?.key_alias || null;

        // Create unique key for aggregation: Date_EntityID_KeyID
        const uniqueKey = `${day.date}_${entityId}_${keyId}`;

        if (!aggregatedData[uniqueKey]) {
          // First time seeing this (Date, Entity ID, Key ID) combination
          aggregatedData[uniqueKey] = {
            Date: day.date,
            entityId,
            entityAlias,
            keyId,
            keyAlias,
            metrics: {
              spend: keyData.metrics?.spend || 0,
              api_requests: keyData.metrics?.api_requests || 0,
              successful_requests: keyData.metrics?.successful_requests || 0,
              failed_requests: keyData.metrics?.failed_requests || 0,
              total_tokens: keyData.metrics?.total_tokens || 0,
              prompt_tokens: keyData.metrics?.prompt_tokens || 0,
              completion_tokens: keyData.metrics?.completion_tokens || 0,
            },
          };
        } else {
          // Aggregate metrics for existing entry
          aggregatedData[uniqueKey].metrics.spend += keyData.metrics?.spend || 0;
          aggregatedData[uniqueKey].metrics.api_requests += keyData.metrics?.api_requests || 0;
          aggregatedData[uniqueKey].metrics.successful_requests += keyData.metrics?.successful_requests || 0;
          aggregatedData[uniqueKey].metrics.failed_requests += keyData.metrics?.failed_requests || 0;
          aggregatedData[uniqueKey].metrics.total_tokens += keyData.metrics?.total_tokens || 0;
          aggregatedData[uniqueKey].metrics.prompt_tokens += keyData.metrics?.prompt_tokens || 0;
          aggregatedData[uniqueKey].metrics.completion_tokens += keyData.metrics?.completion_tokens || 0;
        }
      });
    });
  });

  // Convert aggregated data to array format
  const dailyKeyBreakdown = Object.values(aggregatedData).map((item) => ({
    Date: item.Date,
    [entityLabel]: item.entityAlias,
    [`${entityLabel} ID`]: item.entityId,
    "Key Alias": item.keyAlias || "-",
    "Key ID": item.keyId,
    "Spend ($)": formatNumberWithCommas(item.metrics.spend, 4),
    Requests: item.metrics.api_requests,
    "Successful Requests": item.metrics.successful_requests,
    "Failed Requests": item.metrics.failed_requests,
    "Total Tokens": item.metrics.total_tokens,
    "Prompt Tokens": item.metrics.prompt_tokens,
    "Completion Tokens": item.metrics.completion_tokens,
  }));

  return dailyKeyBreakdown.sort((a, b) => new Date(a.Date).getTime() - new Date(b.Date).getTime());
};

export const generateDailyWithModelsData = (
  spendData: EntitySpendData,
  entityLabel: string,
  teamAliasMap: Record<string, string> = {},
): any[] => {
  const dailyModelBreakdown: any[] = [];

  spendData.results.forEach((day) => {
    const dailyEntityModels: { [key: string]: { [key: string]: any } } = {};

    Object.entries(resolveEntities(day.breakdown)).forEach(([entity, entityData]: [string, any]) => {
      if (!dailyEntityModels[entity]) {
        dailyEntityModels[entity] = {};
      }

      Object.entries(day.breakdown.models || {}).forEach(([model, modelData]: [string, any]) => {
        const apiKeyBreakdown = entityData.api_key_breakdown || {};

        Object.entries(apiKeyBreakdown).forEach(([apiKey, apiKeyData]: [string, any]) => {
          if (!dailyEntityModels[entity][model]) {
            dailyEntityModels[entity][model] = {
              spend: 0,
              requests: 0,
              successful: 0,
              failed: 0,
              tokens: 0,
            };
          }
          dailyEntityModels[entity][model].spend += apiKeyData.metrics.spend || 0;
          dailyEntityModels[entity][model].requests += apiKeyData.metrics.api_requests || 0;
          dailyEntityModels[entity][model].successful += apiKeyData.metrics.successful_requests || 0;
          dailyEntityModels[entity][model].failed += apiKeyData.metrics.failed_requests || 0;
          dailyEntityModels[entity][model].tokens += apiKeyData.metrics.total_tokens || 0;
        });
      });
    });

    Object.entries(dailyEntityModels).forEach(([entity, models]) => {
      const { id, alias } = resolveEntityDisplay(entity, teamAliasMap);

      Object.entries(models).forEach(([model, metrics]: [string, any]) => {
        dailyModelBreakdown.push({
          Date: day.date,
          [entityLabel]: alias,
          [`${entityLabel} ID`]: id,
          Model: model,
          "Spend ($)": formatNumberWithCommas(metrics.spend, 4),
          Requests: metrics.requests,
          Successful: metrics.successful,
          Failed: metrics.failed,
          "Total Tokens": metrics.tokens,
        });
      });
    });
  });

  return dailyModelBreakdown.sort((a, b) => new Date(a.Date).getTime() - new Date(b.Date).getTime());
};

export const generateExportData = (
  spendData: EntitySpendData,
  exportScope: ExportScope,
  entityLabel: string,
  teamAliasMap: Record<string, string> = {},
): any[] => {
  switch (exportScope) {
    case "daily":
      return generateDailyData(spendData, entityLabel, teamAliasMap);
    case "daily_with_keys":
      return generateDailyWithKeysData(spendData, entityLabel, teamAliasMap);
    case "daily_with_models":
      return generateDailyWithModelsData(spendData, entityLabel, teamAliasMap);
    default:
      return generateDailyData(spendData, entityLabel, teamAliasMap);
  }
};

export const generateMetadata = (
  entityType: EntityType,
  dateRange: DateRangePickerValue,
  selectedFilters: string[],
  exportScope: ExportScope,
  spendData: EntitySpendData,
): ExportMetadata => ({
  export_date: new Date().toISOString(),
  entity_type: entityType,
  date_range: {
    from: dateRange.from?.toISOString(),
    to: dateRange.to?.toISOString(),
  },
  filters_applied: selectedFilters.length > 0 ? selectedFilters : "None",
  export_scope: exportScope,
  summary: {
    total_spend: spendData.metadata.total_spend,
    total_requests: spendData.metadata.total_api_requests,
    successful_requests: spendData.metadata.total_successful_requests,
    failed_requests: spendData.metadata.total_failed_requests,
    total_tokens: spendData.metadata.total_tokens,
  },
});

export const handleExportCSV = (
  spendData: EntitySpendData,
  exportScope: ExportScope,
  entityLabel: string,
  entityType: EntityType,
  teamAliasMap: Record<string, string> = {},
): void => {
  const data = generateExportData(spendData, exportScope, entityLabel, teamAliasMap);
  const csv = Papa.unparse(data);
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  const fileName = `${entityType}_usage_${exportScope}_${new Date().toISOString().split("T")[0]}.csv`;
  a.download = fileName;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  window.URL.revokeObjectURL(url);
};

export const handleExportJSON = (
  spendData: EntitySpendData,
  exportScope: ExportScope,
  entityLabel: string,
  entityType: EntityType,
  dateRange: DateRangePickerValue,
  selectedFilters: string[],
  teamAliasMap: Record<string, string> = {},
): void => {
  const data = generateExportData(spendData, exportScope, entityLabel, teamAliasMap);
  const metadata = generateMetadata(entityType, dateRange, selectedFilters, exportScope, spendData);
  const exportObject = {
    metadata,
    data,
  };
  const jsonString = JSON.stringify(exportObject, null, 2);
  const blob = new Blob([jsonString], { type: "application/json" });
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  const fileName = `${entityType}_usage_${exportScope}_${new Date().toISOString().split("T")[0]}.json`;
  a.download = fileName;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  window.URL.revokeObjectURL(url);
};
