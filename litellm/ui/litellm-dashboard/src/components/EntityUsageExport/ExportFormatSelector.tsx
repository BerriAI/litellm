import React from "react";
import { useTranslation } from "react-i18next";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import type { ExportFormat } from "./types";

interface ExportFormatSelectorProps {
  value: ExportFormat;
  onChange: (value: ExportFormat) => void;
}

const ExportFormatSelector: React.FC<ExportFormatSelectorProps> = ({ value, onChange }) => {
  const { t } = useTranslation();
  const labels: Record<ExportFormat, string> = {
    csv: "CSV (Excel, Google Sheets)",
    json: t("usage.export.format.json"),
  };
  return (
    <div>
      <label className="text-sm font-medium text-foreground block mb-2">{t("usage.export.format.label")}</label>
      <Select value={value} onValueChange={(next: ExportFormat | null) => next && onChange(next)}>
        <SelectTrigger className="w-full">
          <SelectValue>{labels[value]}</SelectValue>
        </SelectTrigger>
        <SelectContent>
          {(Object.keys(labels) as ExportFormat[]).map((format) => (
            <SelectItem key={format} value={format}>
              {labels[format]}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  );
};

export default ExportFormatSelector;
