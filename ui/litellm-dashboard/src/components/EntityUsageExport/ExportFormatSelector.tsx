import type { ParseKeys } from "i18next";
import React from "react";
import { useTranslation } from "react-i18next";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import type { ExportFormat } from "./types";

interface ExportFormatSelectorProps {
  value: ExportFormat;
  onChange: (value: ExportFormat) => void;
}

const FORMAT_LABELS: Record<ExportFormat, ParseKeys> = {
  csv: "usage.export.format.csv",
  json: "usage.export.format.json",
};

const ExportFormatSelector: React.FC<ExportFormatSelectorProps> = ({ value, onChange }) => {
  const { t } = useTranslation();
  return (
    <div>
      <label className="text-sm font-medium text-foreground block mb-2">{t("usage.export.format.label")}</label>
      <Select value={value} onValueChange={(next: ExportFormat | null) => next && onChange(next)}>
        <SelectTrigger className="w-full">
          <SelectValue>{t(FORMAT_LABELS[value])}</SelectValue>
        </SelectTrigger>
        <SelectContent>
          {(Object.keys(FORMAT_LABELS) as ExportFormat[]).map((format) => (
            <SelectItem key={format} value={format}>
              {t(FORMAT_LABELS[format])}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  );
};

export default ExportFormatSelector;
