import type { DateRangePickerValue } from "@tremor/react";

export type ExportFormat = "csv" | "json";
export type ExportScope = "daily" | "daily_with_models";
export type EntityType = "tag" | "team" | "organization" | "customer" | "agent";

export interface EntitySpendData {
  results: any[];
  metadata: {
    total_spend: number;
    total_api_requests: number;
    total_successful_requests: number;
    total_failed_requests: number;
    total_tokens: number;
  };
}

export interface EntityUsageExportModalProps {
  isOpen: boolean;
  onClose: () => void;
  entityType: EntityType;
  spendData: EntitySpendData;
  dateRange: DateRangePickerValue;
  selectedFilters: string[];
  customTitle?: string;
}

export interface ExportMetadata {
  export_date: string;
  entity_type: string;
  date_range: {
    from?: string;
    to?: string;
  };
  filters_applied: string[] | string;
  export_scope: ExportScope;
  summary: {
    total_spend: number;
    total_requests: number;
    successful_requests: number;
    failed_requests: number;
    total_tokens: number;
  };
}

export interface EntityBreakdown {
  metrics: {
    spend: number;
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
    api_requests: number;
    successful_requests: number;
    failed_requests: number;
    cache_read_input_tokens: number;
    cache_creation_input_tokens: number;
  };
  metadata: {
    alias: string;
    id: string;
  };
}
