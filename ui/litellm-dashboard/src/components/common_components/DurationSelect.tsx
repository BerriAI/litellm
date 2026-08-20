import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";

interface DurationSelectProps {
  className?: string;
  value?: string;
  onChange?: (value: string, option: { value: string; label: string }) => void;
}

const DURATION_OPTIONS = [
  { value: "24h", label: "Daily" },
  { value: "7d", label: "Weekly" },
  { value: "30d", label: "Monthly" },
];

export default function DurationSelect({ className, value, onChange }: DurationSelectProps) {
  return (
    <Select
      items={DURATION_OPTIONS}
      value={value}
      onValueChange={(nextValue) => {
        const selectedOption = DURATION_OPTIONS.find((option) => option.value === nextValue);
        if (selectedOption) {
          onChange?.(selectedOption.value, selectedOption);
        }
      }}
    >
      <SelectTrigger className={className}>
        <SelectValue placeholder="Select duration" />
      </SelectTrigger>
      <SelectContent>
        {DURATION_OPTIONS.map((option) => (
          <SelectItem key={option.value} value={option.value}>
            {option.label}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}
