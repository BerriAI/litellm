import { formatNumberWithCommas } from "@/utils/dataUtils";
import Papa from "papaparse";
import type { EntitySpendData, EntityBreakdown, ExportMetadata, ExportScope, EntityType } from "./types";
import type { DateRangePickerValue } from "@tremor/react";

export const getEntityBreakdown = (spendData: EntitySpendData): EntityBreakdown[] => {
  const entitySpend: { [key: string]: EntityBreakdown } = {};

  spendData.results.forEach((day) => {
    Object.entries(day.breakdown.entities || {}).forEach(([entity, data]: [string, any]) => {
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
            alias: data.metadata?.team_alias || entity,
            id: entity,
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

export const generateDailyData = (spendData: EntitySpendData, entityLabel: string): any[] => {
  const dailyBreakdown: any[] = [];

  spendData.results.forEach((day) => {
    Object.entries(day.breakdown.entities || {}).forEach(([entity, data]: [string, any]) => {
      dailyBreakdown.push({
        Date: day.date,
        [entityLabel]: data.metadata?.team_alias || entity,
        [`${entityLabel} ID`]: entity,
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

export const generateDailyWithModelsData = (spendData: EntitySpendData, entityLabel: string): any[] => {
  const dailyModelBreakdown: any[] = [];

  spendData.results.forEach((day) => {
    const dailyEntityModels: { [key: string]: { [key: string]: any } } = {};

    Object.entries(day.breakdown.entities || {}).forEach(([entity, entityData]: [string, any]) => {
      const entityName = entityData.metadata?.team_alias || entity;

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
      const entityData = day.breakdown.entities?.[entity];
      const entityName = entityData?.metadata?.team_alias || entity;

      Object.entries(models).forEach(([model, metrics]: [string, any]) => {
        dailyModelBreakdown.push({
          Date: day.date,
          [entityLabel]: entityName,
          [`${entityLabel} ID`]: entity,
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
): any[] => {
  switch (exportScope) {
    case "daily":
      return generateDailyData(spendData, entityLabel);
    case "daily_with_models":
      return generateDailyWithModelsData(spendData, entityLabel);
    default:
      return generateDailyData(spendData, entityLabel);
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
): void => {
  const data = generateExportData(spendData, exportScope, entityLabel);
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
): void => {
  const data = generateExportData(spendData, exportScope, entityLabel);
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
