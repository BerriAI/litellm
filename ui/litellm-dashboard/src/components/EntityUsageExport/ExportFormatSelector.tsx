import React from "react";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import type { ExportFormat } from "./types";
import { useLanguage } from "@/contexts/LanguageContext";

interface ExportFormatSelectorProps {
  value: ExportFormat;
  onChange: (value: ExportFormat) => void;
}

const FORMAT_LABELS: Record<ExportFormat, string> = {
  csv: "CSV (Excel, Google Sheets)",
  json: "JSON (includes metadata)",
};

const ExportFormatSelector: React.FC<ExportFormatSelectorProps> = ({ value, onChange }) => {
  const { t } = useLanguage();
  return (
    <div>
      <label className="text-sm font-medium text-foreground block mb-2">{t("Format")}</label>
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
