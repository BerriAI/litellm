import { useLanguage } from "@/contexts/LanguageContext";
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
  const { t } = useLanguage();
  return (
    <SearchSelect
      aria-label={t("View client usage")}
      placeholder={t("View client usage")}
      value={value}
      options={CLIENT_OPTIONS}
      onValueChange={(client) => onChange(parseUsageClient(client))}
    />
  );
}
