import React from "react";
import { useTranslation } from "react-i18next";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import type { UsageExportFormat } from "./types";

interface UsageExportFormatSelectorProps {
  value: UsageExportFormat;
  onChange: (value: UsageExportFormat) => void;
}

const UsageExportFormatSelector: React.FC<UsageExportFormatSelectorProps> = ({ value, onChange }) => {
  const { t } = useTranslation();
  const labels: Record<UsageExportFormat, string> = {
    csv: "CSV (Excel, Google Sheets)",
    json: t("usage.export.format.json"),
    pdf: t("usage.export.format.pdf"),
  };
  return (
    <div>
      <label className="text-sm font-medium text-foreground block mb-2">{t("usage.export.format.label")}</label>
      <Select value={value} onValueChange={(next: UsageExportFormat | null) => next && onChange(next)}>
        <SelectTrigger className="w-full">
          <SelectValue>{labels[value]}</SelectValue>
        </SelectTrigger>
        <SelectContent>
          {(Object.keys(labels) as UsageExportFormat[]).map((format) => (
            <SelectItem key={format} value={format}>
              {labels[format]}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  );
};

export default UsageExportFormatSelector;
