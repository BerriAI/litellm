import { SearchSelect } from "@/components/shared/SearchSelect";
import type { UsageClient } from "../hooks/clientUsageData";

const CLIENT_OPTIONS = [
  { value: "claude-desktop", label: "Claude Desktop" },
  { value: "claude-code", label: "Claude Code" },
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
      onValueChange={(client) => onChange(client === "claude-desktop" || client === "claude-code" ? client : null)}
    />
  );
}
