import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Loader2 } from "lucide-react";
import { SelectorOption, EndpointConfig } from "../endpoint_config";

interface UnifiedSelectorProps {
  value: string;
  options: SelectorOption[];
  loading: boolean;
  config: EndpointConfig;
  onChange: (value: string) => void;
}

export function UnifiedSelector({
  value,
  options,
  loading,
  config,
  onChange,
}: UnifiedSelectorProps) {
  return (
    <Select value={value || undefined} onValueChange={onChange}>
      <SelectTrigger className="w-48 md:w-64 lg:w-72">
        <SelectValue
          placeholder={
            loading
              ? `Loading ${config.selectorLabel.toLowerCase()}s...`
              : config.selectorPlaceholder
          }
        />
      </SelectTrigger>
      <SelectContent>
        {loading ? (
          <div className="flex items-center justify-center py-2">
            <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
          </div>
        ) : options.length === 0 ? (
          <div className="py-2 px-3 text-sm text-muted-foreground">
            No {config.selectorLabel.toLowerCase()}s available
          </div>
        ) : (
          options.map((opt) => (
            <SelectItem key={opt.value as string} value={opt.value as string}>
              {opt.label}
            </SelectItem>
          ))
        )}
      </SelectContent>
    </Select>
  );
}
