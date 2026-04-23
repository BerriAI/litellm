import React from "react";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import type { ExportFormat } from "./types";

interface ExportFormatSelectorProps {
  value: ExportFormat;
  onChange: (value: ExportFormat) => void;
}

const ExportFormatSelector: React.FC<ExportFormatSelectorProps> = ({
  value,
  onChange,
}) => {
  return (
    <div>
      <Label className="text-sm font-medium block mb-2">Format</Label>
      <Select value={value} onValueChange={(v) => onChange(v as ExportFormat)}>
        <SelectTrigger className="w-full">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="csv">CSV (Excel, Google Sheets)</SelectItem>
          <SelectItem value="json">JSON (includes metadata)</SelectItem>
        </SelectContent>
      </Select>
    </div>
  );
};

export default ExportFormatSelector;
