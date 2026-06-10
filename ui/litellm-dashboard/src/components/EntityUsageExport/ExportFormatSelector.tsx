import React, { useMemo } from "react";
import { useTranslation } from "react-i18next";
import type { TFunction } from "i18next";
import { Select } from "antd";
import type { ExportFormat } from "./types";

interface ExportFormatSelectorProps {
  value: ExportFormat;
  onChange: (value: ExportFormat) => void;
}

const getFormatOptions = (t: TFunction) => [
  {
    value: "csv",
    label: t("usageExport.exportFormatSelector.optionCsv"),
  },
  {
    value: "json",
    label: t("usageExport.exportFormatSelector.optionJson"),
  },
];

const ExportFormatSelector: React.FC<ExportFormatSelectorProps> = ({ value, onChange }) => {
  const { t } = useTranslation();
  const formatOptions = useMemo(() => getFormatOptions(t), [t]);

  return (
    <div>
      <label className="text-sm font-medium text-gray-700 block mb-2">
        {t("usageExport.exportFormatSelector.label")}
      </label>
      <Select value={value} onChange={onChange} className="w-full" options={formatOptions} />
    </div>
  );
};

export default ExportFormatSelector;
