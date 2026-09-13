import React from "react";
import type { DateRangePickerValue } from "@/components/shared/date_picker_types";

interface ExportSummaryProps {
  dateRange: DateRangePickerValue;
  selectedFilters: string[];
}

const ExportSummary: React.FC<ExportSummaryProps> = ({ dateRange, selectedFilters }) => {
  return (
    <div className="text-sm text-muted-foreground">
      {dateRange.from?.toLocaleDateString()} - {dateRange.to?.toLocaleDateString()}
      {selectedFilters.length > 0 && ` · ${selectedFilters.length} filter${selectedFilters.length > 1 ? "s" : ""}`}
    </div>
  );
};

export default ExportSummary;
