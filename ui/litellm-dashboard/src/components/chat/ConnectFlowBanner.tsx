"use client";

import React from "react";
import { CheckCircle } from "lucide-react";
import { getProxyBaseUrl } from "@/components/networking";

interface Props {
  flowHandle: string;
  clientOrigin: string | null;
}

/**
 * The interlude shown when a DCR client (Claude Desktop, MCP Inspector) sends the user
 * through the gateway sign-in and lands them on the apps grid to authorize servers. The
 * grid below authorizes individual servers into the per-user vault; this banner is the
 * deliberate finish step.
 *
 * "Finish connecting" is a native form POST to the proxy's /authorize/complete, not a
 * fetch: the endpoint 303-redirects the browser back to the DCR client's own redirect URI
 * with the gateway authorization code, and only a full-page navigation carries the
 * HttpOnly per-flow cookie and follows that cross-origin redirect. The flow handle is the
 * only field; the sealed flow cookie set at /authorize holds everything else.
 */
const ConnectFlowBanner: React.FC<Props> = ({ flowHandle, clientOrigin }) => {
  const action = `${getProxyBaseUrl()}/authorize/complete`;
  const clientLabel = clientOrigin ?? "the application";

  return (
    <div className="mb-6 rounded-lg border border-primary/30 bg-primary/5 px-5 py-4">
      <div className="flex items-start justify-between gap-4 flex-wrap">
        <div className="flex items-start gap-3 min-w-0">
          <CheckCircle className="h-5 w-5 text-primary shrink-0 mt-0.5" />
          <div className="min-w-0">
            <p className="text-sm font-semibold text-foreground">Connect your MCP servers to {clientLabel}</p>
            <p className="text-[13px] text-muted-foreground mt-0.5">
              Authorize the servers you want to use below. When you are ready, finish connecting and you will be
              returned to {clientLabel}.
            </p>
          </div>
        </div>
        <form method="POST" action={action} className="shrink-0">
          <input type="hidden" name="flow" value={flowHandle} />
          <button
            type="submit"
            className="h-[38px] rounded-md bg-primary px-4 text-sm font-semibold text-primary-foreground hover:bg-primary/90"
          >
            Finish connecting
          </button>
        </form>
      </div>
    </div>
  );
};

export default ConnectFlowBanner;
