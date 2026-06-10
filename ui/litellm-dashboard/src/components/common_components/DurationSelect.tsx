import { Select } from "antd";
import { useTranslation } from "react-i18next";

interface DurationSelectProps {
  className?: string;
  value?: string;
  onChange?: (value: string) => void;
}

export default function DurationSelect({ className, value, onChange }: DurationSelectProps) {
  const { t } = useTranslation();

  return (
    <Select className={className} value={value} onChange={onChange}>
      <Select.Option value="24h">{t("commonComponents.durationSelect.daily")}</Select.Option>
      <Select.Option value="7d">{t("commonComponents.durationSelect.weekly")}</Select.Option>
      <Select.Option value="30d">{t("commonComponents.durationSelect.monthly")}</Select.Option>
    </Select>
  );
}
