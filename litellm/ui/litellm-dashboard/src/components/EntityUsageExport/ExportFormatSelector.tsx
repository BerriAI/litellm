import type { ParseKeys } from "i18next";
import React from "react";
import { useTranslation } from "react-i18next";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import type { UsageExportFormat } from "./types";

interface UsageExportFormatSelectorProps {
  value: UsageExportFormat;
  onChange: (value: UsageExportFormat) => void;
}

const FORMAT_LABELS: Record<UsageExportFormat, ParseKeys> = {
  csv: "usage.export.format.csv",
  json: "usage.export.format.json",
  pdf: "usage.export.format.pdf",
};

const UsageExportFormatSelector: React.FC<UsageExportFormatSelectorProps> = ({ value, onChange }) => {
  const { t } = useTranslation();
  return (
    <div>
      <label className="text-sm font-medium text-foreground block mb-2">{t("usage.export.format.label")}</label>
      <Select value={value} onValueChange={(next: UsageExportFormat | null) => next && onChange(next)}>
        <SelectTrigger className="w-full">
          <SelectValue>{t(FORMAT_LABELS[value])}</SelectValue>
        </SelectTrigger>
        <SelectContent>
          {(Object.keys(FORMAT_LABELS) as UsageExportFormat[]).map((format) => (
            <SelectItem key={format} value={format}>
              {t(FORMAT_LABELS[format])}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  );
};

export default UsageExportFormatSelector;
