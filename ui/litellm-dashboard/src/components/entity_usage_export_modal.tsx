import React, { useState } from "react";
import { Text, Button } from "@tremor/react";
import { Modal, Radio, Select } from "antd";
import Papa from "papaparse";
import NotificationsManager from "./molecules/notifications_manager";
import { formatNumberWithCommas } from "@/utils/dataUtils";
import type { DateRangePickerValue } from "@tremor/react";

interface EntitySpendData {
  results: any[];
  metadata: {
    total_spend: number;
    total_api_requests: number;
    total_successful_requests: number;
    total_failed_requests: number;
    total_tokens: number;
  };
}

interface EntityUsageExportModalProps {
  isOpen: boolean;
  onClose: () => void;
  entityType: "tag" | "team";
  spendData: EntitySpendData;
  dateRange: DateRangePickerValue;
  selectedFilters: string[];
}

type ExportFormat = "csv" | "json";
type ExportScope = "daily" | "daily_with_models";

const EntityUsageExportModal: React.FC<EntityUsageExportModalProps> = ({
  isOpen,
  onClose,
  entityType,
  spendData,
  dateRange,
  selectedFilters,
}) => {
  const [exportFormat, setExportFormat] = useState<ExportFormat>("csv");
  const [exportScope, setExportScope] = useState<ExportScope>("daily");
  const [isExporting, setIsExporting] = useState(false);

  const entityLabel = entityType === "tag" ? "Tag" : "Team";
  const entityLabelPlural = entityType === "tag" ? "Tags" : "Teams";

  const getEntityBreakdown = () => {
    const entitySpend: { [key: string]: any } = {};
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

    return Object.values(entitySpend).sort((a: any, b: any) => b.metrics.spend - a.metrics.spend);
  };

  const generateDailyData = () => {
    const dailyBreakdown: any[] = [];

    spendData.results.forEach((day) => {
      Object.entries(day.breakdown.entities || {}).forEach(([entity, data]: [string, any]) => {
        dailyBreakdown.push({
          Date: day.date,
          [entityLabel]: data.metadata?.team_alias || entity,
          [`${entityLabel} ID`]: entity,
          "Spend ($)": formatNumberWithCommas(data.metrics.spend, 4),
          "Requests": data.metrics.api_requests,
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

  const generateDailyWithModelsData = () => {
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
            "Requests": metrics.requests,
            "Successful": metrics.successful,
            "Failed": metrics.failed,
            "Total Tokens": metrics.tokens,
          });
        });
      });
    });

    return dailyModelBreakdown.sort((a, b) => new Date(a.Date).getTime() - new Date(b.Date).getTime());
  };

  const generateExportData = () => {
    switch (exportScope) {
      case "daily":
        return generateDailyData();
      case "daily_with_models":
        return generateDailyWithModelsData();
      default:
        return generateDailyData();
    }
  };

  const generateMetadata = () => ({
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

  const handleExportCSV = () => {
    const data = generateExportData();
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

  const handleExportJSON = () => {
    const data = generateExportData();
    const metadata = generateMetadata();
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

  const handleExport = async (format?: ExportFormat) => {
    const formatToUse = format || exportFormat;
    setIsExporting(true);
    try {
      if (formatToUse === "csv") {
        handleExportCSV();
        NotificationsManager.success(`${entityLabel} usage data exported successfully as CSV`);
      } else {
        handleExportJSON();
        NotificationsManager.success(`${entityLabel} usage data exported successfully as JSON`);
      }
      onClose();
    } catch (error) {
      console.error("Error exporting data:", error);
      NotificationsManager.fromBackend("Failed to export data");
    } finally {
      setIsExporting(false);
    }
  };

  const getScopeDescription = () => {
    switch (exportScope) {
      case "daily":
        return `Day-by-day breakdown for each ${entityType}`;
      case "daily_with_models":
        return `Day-by-day breakdown by ${entityType} and model`;
      default:
        return "";
    }
  };

  const getRecordCount = () => {
    const data = generateExportData();
    return data.length;
  };

  return (
    <Modal
      title={<span className="text-base font-semibold">Export {entityLabel} Usage</span>}
      open={isOpen}
      onCancel={onClose}
      footer={null}
      width={480}
      destroyOnClose
    >
      <div className="space-y-5 py-2">
        <div className="text-sm text-gray-500">
          {dateRange.from?.toLocaleDateString()} - {dateRange.to?.toLocaleDateString()}
          {selectedFilters.length > 0 && ` · ${selectedFilters.length} filter${selectedFilters.length > 1 ? "s" : ""}`}
        </div>

        <div>
          <label className="text-sm font-medium text-gray-700 block mb-2">Export type</label>
          <Radio.Group
            value={exportScope}
            onChange={(e) => setExportScope(e.target.value)}
            className="w-full"
          >
            <div className="space-y-2">
              <label className="flex items-start p-3 border border-gray-200 rounded-lg hover:bg-gray-50 cursor-pointer transition-colors">
                <Radio value="daily" className="mt-0.5" />
                <div className="ml-3 flex-1">
                  <div className="font-medium text-sm">Day-by-day breakdown</div>
                  <div className="text-xs text-gray-500 mt-0.5">Daily metrics for each {entityType}</div>
                </div>
              </label>
              
              <label className="flex items-start p-3 border border-gray-200 rounded-lg hover:bg-gray-50 cursor-pointer transition-colors">
                <Radio value="daily_with_models" className="mt-0.5" />
                <div className="ml-3 flex-1">
                  <div className="font-medium text-sm">Day-by-day by {entityType} and model</div>
                  <div className="text-xs text-gray-500 mt-0.5">Daily metrics split by model</div>
                </div>
              </label>
            </div>
          </Radio.Group>
        </div>

        <div>
          <label className="text-sm font-medium text-gray-700 block mb-2">Format</label>
          <Select
            value={exportFormat}
            onChange={setExportFormat}
            className="w-full"
            options={[
              {
                value: "csv",
                label: "CSV (Excel, Google Sheets)",
              },
              {
                value: "json",
                label: "JSON (includes metadata)",
              },
            ]}
          />
        </div>

        <div className="flex items-center justify-end gap-2 pt-4 border-t">
          <Button variant="secondary" onClick={onClose} disabled={isExporting} size="sm">
            Cancel
          </Button>
          <Button 
            onClick={() => handleExport()} 
            loading={isExporting} 
            disabled={isExporting}
            size="sm"
          >
            {isExporting ? "Exporting..." : `Export ${exportFormat.toUpperCase()}`}
          </Button>
        </div>
      </div>
    </Modal>
  );
};

export default EntityUsageExportModal;

