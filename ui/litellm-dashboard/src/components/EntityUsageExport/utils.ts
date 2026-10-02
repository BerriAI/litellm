import type { DateRangePickerValue } from "@/components/shared/date_picker_types";
import type { EntityType, ExportFormat, ExportType } from "./types";

const fileDay = (date: Date | undefined): string =>
  date
    ? `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, "0")}-${String(date.getDate()).padStart(2, "0")}`
    : "all";

export const exportFilename = (
  entityType: EntityType,
  exportType: ExportType,
  format: ExportFormat,
  dateRange: DateRangePickerValue,
): string => `${entityType}_usage_${exportType}_${fileDay(dateRange.from)}_${fileDay(dateRange.to)}.${format}`;

export const downloadBlob = (blob: Blob, filename: string): void => {
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  window.URL.revokeObjectURL(url);
};
