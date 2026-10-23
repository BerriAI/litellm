import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { OpenAIUpstreamFormat } from "./openai_upstream_format";

const options = [
  { value: OpenAIUpstreamFormat.Default, label: "Default" },
  { value: OpenAIUpstreamFormat.SystemOne, label: "System One" },
  { value: OpenAIUpstreamFormat.Decisions, label: "OpenAI Decisions" },
];

const descriptions: Record<OpenAIUpstreamFormat, string> = {
  [OpenAIUpstreamFormat.Default]: "Use the provider's default format for decision requests",
  [OpenAIUpstreamFormat.SystemOne]: "Send decision requests to /v1/systemone. Chat completions remain available",
  [OpenAIUpstreamFormat.Decisions]: "Send decision requests to /v1/decisions. Chat completions remain available",
};

interface UpstreamFormatSelectProps {
  id: string;
  value: OpenAIUpstreamFormat;
  onChange: (value: OpenAIUpstreamFormat) => void;
  onBlur?: () => void;
}

export default function UpstreamFormatSelect({ id, value, onChange, onBlur }: UpstreamFormatSelectProps) {
  return (
    <div className="space-y-2">
      <Select
        items={options}
        value={value}
        onValueChange={(selected) => {
          if (selected !== null) onChange(selected);
        }}
      >
        <SelectTrigger
          id={id}
          aria-label="Upstream format"
          aria-describedby={`${id}-help`}
          onBlur={onBlur}
          className="w-full"
        >
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {options.map((option) => (
            <SelectItem key={option.value} value={option.value}>
              {option.label}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      <p id={`${id}-help`} className="text-xs text-muted-foreground">
        {descriptions[value]}
      </p>
    </div>
  );
}
