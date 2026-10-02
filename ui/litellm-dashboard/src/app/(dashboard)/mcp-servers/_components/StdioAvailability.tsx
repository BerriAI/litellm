import { type FC } from "react";
import { TriangleAlert } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/shared/Alert";
import { SelectItem } from "@/components/ui/select";
import { SimpleTooltip } from "@/components/ui/tooltip";
import { TRANSPORT, TRANSPORT_ITEMS } from "@/components/mcp_tools/types";
import { $api } from "@/lib/http/api";

export const STDIO_DISABLED_MESSAGE =
  "stdio MCP servers are disabled on this proxy. Set LITELLM_ENABLE_MCP_STDIO=true on the proxy and restart to enable them";

export const useMcpStdioEnabled = (): boolean =>
  $api.useQuery("get", "/.well-known/litellm-ui-config").data?.mcp_stdio_enabled !== false;

export const TransportSelectItems: FC<{ stdioEnabled: boolean }> = ({ stdioEnabled }) => (
  <>
    {TRANSPORT_ITEMS.map((item) => {
      const disabled = item.value === TRANSPORT.STDIO && !stdioEnabled;
      return (
        <SelectItem key={item.value} value={item.value} disabled={disabled}>
          {item.label}
          {disabled && <SimpleTooltip content={STDIO_DISABLED_MESSAGE} className="pointer-events-auto" />}
        </SelectItem>
      );
    })}
  </>
);

export const StdioDisabledBanner: FC = () => (
  <Alert variant="warning" className="mb-4 rounded-lg">
    <TriangleAlert />
    <AlertTitle>stdio is disabled on this proxy</AlertTitle>
    <AlertDescription>
      {STDIO_DISABLED_MESSAGE}. Until then this server cannot start or be saved as stdio.
    </AlertDescription>
  </Alert>
);
