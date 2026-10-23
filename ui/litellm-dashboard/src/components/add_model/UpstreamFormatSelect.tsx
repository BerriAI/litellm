import { ConfigProvider, Select, theme } from "antd";
import { useTheme } from "next-themes";
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
  const { resolvedTheme } = useTheme();
  return (
    <div className="space-y-2">
      <ConfigProvider theme={{ algorithm: resolvedTheme === "dark" ? theme.darkAlgorithm : theme.defaultAlgorithm }}>
        <Select<OpenAIUpstreamFormat>
          id={id}
          aria-label="Upstream format"
          aria-describedby={`${id}-help`}
          value={value}
          onChange={onChange}
          onBlur={onBlur}
          options={options}
          virtual={false}
          size="large"
          className="w-full"
        />
      </ConfigProvider>
      <p id={`${id}-help`} className="text-xs text-muted-foreground">
        {descriptions[value]}
      </p>
    </div>
  );
}
