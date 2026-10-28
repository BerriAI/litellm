import { SearchSelect } from "@/components/shared/SearchSelect";
import { parseUsageClient, UsageClient } from "./overview/agentCatalog";

const CLIENT_OPTIONS = [
  { value: UsageClient.ClaudeDesktop, label: "Claude Desktop" },
  { value: UsageClient.ClaudeCode, label: "Claude Code" },
];

export default function ClientUsageSelect({
  value,
  onChange,
}: {
  value: UsageClient | null;
  onChange: (client: UsageClient | null) => void;
}) {
  return (
    <SearchSelect
      aria-label="View client usage"
      placeholder="View client usage"
      value={value}
      options={CLIENT_OPTIONS}
      onValueChange={(client) => onChange(parseUsageClient(client))}
    />
  );
}
