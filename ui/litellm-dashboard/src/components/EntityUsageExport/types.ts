import type { DateRangePickerValue } from "@/components/shared/date_picker_types";
import type { Team } from "@/components/key_team_helpers/key_list";
import type { DailyActivityEntity, ExportFormat, ExportType } from "@/components/UsagePage/dailyActivityApi";

export type { ExportFormat, ExportType };
export type EntityType = DailyActivityEntity;

export interface EntityUsageExportModalProps {
  isOpen: boolean;
  onClose: () => void;
  entityType: EntityType;
  onExport: (exportType: ExportType, format: ExportFormat) => Promise<Blob>;
  dateRange: DateRangePickerValue;
  selectedFilters?: string[];
  customTitle?: string;
  teams?: Team[];
}
