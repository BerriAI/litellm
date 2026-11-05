import type { ParseKeys } from "i18next";
import { type FC } from "react";
import { useTranslation } from "react-i18next";
import { TriangleAlert } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/shared/Alert";
import { SelectItem } from "@/components/ui/select";
import { SimpleTooltip } from "@/components/ui/tooltip";
import { TRANSPORT, TRANSPORT_ITEMS } from "@/components/mcp_tools/types";
import { $api } from "@/lib/http/api";
import en from "@/i18n/locales/en.json";

export const STDIO_DISABLED_MESSAGE_KEY: ParseKeys = "mcpServers.stdioAvailability.disabledMessage";

export const STDIO_DISABLED_MESSAGE = en.mcpServers.stdioAvailability.disabledMessage;

export const useMcpStdioEnabled = (): boolean =>
  $api.useQuery("get", "/.well-known/litellm-ui-config").data?.mcp_stdio_enabled !== false;

export const TransportSelectItems: FC<{ stdioEnabled: boolean }> = ({ stdioEnabled }) => {
  const { t } = useTranslation();
  return (
    <>
      {TRANSPORT_ITEMS.map((item) => {
        const disabled = item.value === TRANSPORT.STDIO && !stdioEnabled;
        return (
          <SelectItem key={item.value} value={item.value} disabled={disabled}>
            {item.label}
            {disabled && <SimpleTooltip content={t(STDIO_DISABLED_MESSAGE_KEY)} className="pointer-events-auto" />}
          </SelectItem>
        );
      })}
    </>
  );
};

export const StdioDisabledBanner: FC = () => {
  const { t } = useTranslation();
  return (
    <Alert variant="warning" className="mb-4 rounded-lg">
      <TriangleAlert />
      <AlertTitle>{t("mcpServers.stdioAvailability.banner.title")}</AlertTitle>
      <AlertDescription>
        {t("mcpServers.stdioAvailability.banner.description", { disabledMessage: t(STDIO_DISABLED_MESSAGE_KEY) })}
      </AlertDescription>
    </Alert>
  );
};
