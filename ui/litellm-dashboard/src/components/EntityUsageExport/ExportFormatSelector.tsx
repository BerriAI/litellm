import React from "react";
import { Select } from "antd";
import type { ExportFormat } from "./types";

interface ExportFormatSelectorProps {
  value: ExportFormat;
  onChange: (value: ExportFormat) => void;
}

const ExportFormatSelector: React.FC<ExportFormatSelectorProps> = ({ value, onChange }) => {
  return (
    <div>
      <label className="text-sm font-medium text-gray-700 block mb-2">Format</label>
      <Select
        value={value}
        onChange={onChange}
        className="w-full"
        options={[
          {
            value: "csv",
            label: "CSV (Excel, Google Sheets)",
          },
          {
            value: "json",
            label: "JSON (includes metadata)",
          },
        ]}
      />
    </div>
  );
};

export default ExportFormatSelector;

