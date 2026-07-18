"use client";

import React, { useEffect, useRef } from "react";
import { CheckCircle } from "lucide-react";
import { getProxyBaseUrl } from "@/components/networking";
import { PERSERVER_CONNECTING_KEY } from "@/hooks/mcpOAuthUtils";

interface Props {
  flowHandle: string;
  clientOrigin: string | null;
}

/**
 * The interlude shown when a DCR client (Claude Desktop, MCP Inspector) sends the user
 * through the gateway sign-in and lands them on the apps grid to authorize servers. The
 * grid below authorizes individual servers into the per-user vault; this banner is the
 * finish step that returns the user to the client.
 *
 * Finishing happens two ways, both hitting the proxy's /authorize/complete, which mints the
 * gateway authorization code and 303-redirects to the DCR client's own redirect URI:
 *  - The explicit "Finish connecting" button is a native form POST, so the full-page
 *    navigation carries the HttpOnly per-flow cookie and follows the cross-origin redirect
 *    to the client's loopback. This is the reliable path.
 *  - Closing (or navigating away from) the tab fires a best-effort navigator.sendBeacon to the
 *    same endpoint. The browser follows the 303 to the client's loopback, so in most browsers
 *    the code still reaches the client without an explicit click. This is a convenience, not a
 *    consent gate: consent already happened at sign-in, so returning the user is safe. It is
 *    skipped while a per-server connect is navigating away (that is not leaving the flow),
 *    and after the button was pressed (which already delivers the code).
 */
const ConnectFlowBanner: React.FC<Props> = ({ flowHandle, clientOrigin }) => {
  const action = `${getProxyBaseUrl()}/authorize/complete`;
  const clientLabel = clientOrigin ?? "the application";
  const finishedRef = useRef(false);

  useEffect(() => {
    sessionStorage.removeItem(PERSERVER_CONNECTING_KEY);

    const autoFinishOnLeave = () => {
      if (finishedRef.current) return;
      if (sessionStorage.getItem(PERSERVER_CONNECTING_KEY) === "1") return;
      if (typeof navigator.sendBeacon === "function") {
        navigator.sendBeacon(action, new URLSearchParams({ flow: flowHandle }));
      }
    };
    window.addEventListener("pagehide", autoFinishOnLeave);
    return () => window.removeEventListener("pagehide", autoFinishOnLeave);
  }, [action, flowHandle]);

  return (
    <div className="mb-6 rounded-lg border border-primary/30 bg-primary/5 px-5 py-4">
      <div className="flex items-start justify-between gap-4 flex-wrap">
        <div className="flex items-start gap-3 min-w-0">
          <CheckCircle className="h-5 w-5 text-primary shrink-0 mt-0.5" />
          <div className="min-w-0">
            <p className="text-sm font-semibold text-foreground">Connect your MCP servers to {clientLabel}</p>
            <p className="text-[13px] text-muted-foreground mt-0.5">
              Authorize the servers you want to use below, then finish connecting to return to {clientLabel}. Closing
              this tab finishes for you.
            </p>
          </div>
        </div>
        <form method="POST" action={action} className="shrink-0" onSubmit={() => (finishedRef.current = true)}>
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
