import { Select } from "antd";

interface DurationSelectProps {
  className?: string;
  value?: string;
  onChange?: (value: string) => void;
  disabled?: boolean;
}

export default function DurationSelect({ className, value, onChange, disabled = false }: DurationSelectProps) {
  return (
    <Select className={className} value={value} onChange={onChange} disabled={disabled}>
      <Select.Option value="24h">Daily</Select.Option>
      <Select.Option value="7d">Weekly</Select.Option>
      <Select.Option value="30d">Monthly</Select.Option>
    </Select>
  );
}
