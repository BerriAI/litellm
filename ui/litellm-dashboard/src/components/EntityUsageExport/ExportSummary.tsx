import React from "react";
import type { DateRangePickerValue } from "@/components/shared/date_picker_types";
import { useLanguage } from "@/contexts/LanguageContext";

interface ExportSummaryProps {
  dateRange: DateRangePickerValue;
  selectedFilters: readonly string[];
}

const ExportSummary: React.FC<ExportSummaryProps> = ({ dateRange, selectedFilters }) => {
  const { t, language } = useLanguage();
  const locale = language === "en" ? undefined : language;
  return (
    <div className="text-sm text-muted-foreground">
      {dateRange.from?.toLocaleDateString(locale)} - {dateRange.to?.toLocaleDateString(locale)}
      {selectedFilters.length > 0 &&
        t(selectedFilters.length === 1 ? " · {count} filter" : " · {count} filters", { count: selectedFilters.length })}
    </div>
  );
};

export default ExportSummary;
