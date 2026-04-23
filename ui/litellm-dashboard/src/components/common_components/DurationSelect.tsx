import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { cn } from "@/lib/utils";

interface DurationSelectProps {
  className?: string;
  value?: string;
  onChange?: (value: string) => void;
}

export default function DurationSelect({
  className,
  value,
  onChange,
}: DurationSelectProps) {
  return (
    <Select value={value} onValueChange={(v) => onChange?.(v)}>
      <SelectTrigger className={cn(className)}>
        <SelectValue />
      </SelectTrigger>
      <SelectContent>
        <SelectItem value="24h">Daily</SelectItem>
        <SelectItem value="7d">Weekly</SelectItem>
        <SelectItem value="30d">Monthly</SelectItem>
      </SelectContent>
    </Select>
  );
}
