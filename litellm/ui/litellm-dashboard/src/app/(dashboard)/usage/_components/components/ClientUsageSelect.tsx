import { useTranslation } from "react-i18next";
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
  const { t } = useTranslation();
  return (
    <SearchSelect
      aria-label={t("usage.clientSelect.label")}
      placeholder={t("usage.clientSelect.placeholder")}
      value={value}
      options={CLIENT_OPTIONS}
      onValueChange={(client) => onChange(parseUsageClient(client))}
    />
  );
}
