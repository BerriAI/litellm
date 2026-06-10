import React from "react";
import { useTranslation } from "react-i18next";
import type { DateRangePickerValue } from "@tremor/react";

interface ExportSummaryProps {
  dateRange: DateRangePickerValue;
  selectedFilters: string[];
}

const ExportSummary: React.FC<ExportSummaryProps> = ({ dateRange, selectedFilters }) => {
  const { t } = useTranslation();

  return (
    <div className="text-sm text-gray-500">
      {dateRange.from?.toLocaleDateString()} - {dateRange.to?.toLocaleDateString()}
      {selectedFilters.length > 0 &&
        ` · ${t("usageExport.exportSummary.filterCount", { count: selectedFilters.length })}`}
    </div>
  );
};

export default ExportSummary;
