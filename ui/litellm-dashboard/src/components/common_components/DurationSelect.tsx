import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";

interface DurationSelectProps {
  className?: string;
  value?: string;
  onChange?: (value: string) => void;
}

export default function DurationSelect({ className, value, onChange }: DurationSelectProps) {
  return (
    <Select
      value={value}
      onValueChange={(nextValue) => {
        if (nextValue !== null) {
          onChange?.(nextValue);
        }
      }}
    >
      <SelectTrigger className={className}>
        <SelectValue placeholder="Select duration" />
      </SelectTrigger>
      <SelectContent>
        <SelectItem value="24h">Daily</SelectItem>
        <SelectItem value="7d">Weekly</SelectItem>
        <SelectItem value="30d">Monthly</SelectItem>
      </SelectContent>
    </Select>
  );
}
