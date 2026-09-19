"use client";

import React from "react";
import { CheckCircle } from "lucide-react";
import { getProxyBaseUrl, ConnectFlowStatus } from "@/components/networking";
import { OAuth2ConnectButton } from "@/components/chat/MCPAppsPanel";

interface Props {
  flowHandle: string;
  flow?: ConnectFlowStatus;
  accessToken: string;
  onConnected: () => void;
  failed: boolean;
}

/** Finish remains an explicit POST because a cross-site navigation must never mint a code. */
export function isLoopbackOrigin(origin: string | null): boolean {
  if (!origin) return false;
  try {
    const hostname = new URL(origin).hostname.replace(/^\[|\]$/g, "");
    return hostname === "localhost" || hostname === "::1" || /^127(\.\d{1,3}){3}$/.test(hostname);
  } catch {
    return false;
  }
}

const copyFor = (flow: ConnectFlowStatus | undefined, failed: boolean): readonly [string, string] => {
  const clientLabel = flow?.client_origin ?? "the application";
  const serverLabel = flow?.server_name ?? "the requested MCP server";
  if (failed || flow === undefined || flow.state === "stale") {
    return [
      "The connection cannot continue",
      `The gateway could not validate this connection. Cancel to return to ${clientLabel}.`,
    ];
  }
  if (flow.state === "unscoped") {
    return [
      `Connect your MCP servers to ${clientLabel}`,
      `Authorize the servers you want to use below, then click Finish connecting to return to ${clientLabel}.`,
    ];
  }
  if (flow.state === "interactive" && !flow.connected) {
    return [
      `Allow ${clientLabel} to use ${serverLabel}`,
      `Authorize ${serverLabel} below to continue, or cancel to send ${clientLabel} away.`,
    ];
  }
  return [
    `Allow ${clientLabel} to use ${serverLabel}`,
    `Click Finish connecting to give ${clientLabel} access to ${serverLabel} as you.`,
  ];
};

const ConnectFlowBanner: React.FC<Props> = ({ flowHandle, flow, accessToken, onConnected, failed }) => {
  const action = `${getProxyBaseUrl()}/authorize/complete`;
  const state = failed || flow === undefined ? "stale" : flow.state;
  const canFinish = state === "unscoped" || (state !== "stale" && flow?.connected === true);
  const canCancel = state !== "unscoped";
  const loopbackClient = isLoopbackOrigin(flow?.client_origin ?? null);
  const vendorServer =
    state === "interactive" && flow?.connected === false && flow.server_id !== null
      ? { server_id: flow.server_id, server_name: flow.server_name }
      : null;
  const copy = copyFor(flow, failed);

  return (
    <div className="mb-6 rounded-lg border border-primary/30 bg-primary/5 px-5 py-4">
      <div className="flex items-start justify-between gap-4 flex-wrap">
        <div className="flex items-start gap-3 min-w-0">
          <CheckCircle className="h-5 w-5 text-primary shrink-0 mt-0.5" />
          <div className="min-w-0">
            <p className="text-sm font-semibold text-foreground">{copy[0]}</p>
            <p className="text-[13px] text-muted-foreground mt-0.5">{copy[1]}</p>
          </div>
        </div>
        <div className="flex shrink-0 gap-2">
          {vendorServer !== null && (
            <OAuth2ConnectButton
              server={vendorServer}
              accessToken={accessToken}
              onConnect={onConnected}
              variant="button"
              autoStartKey={`litellm-mcp-autostart:${flowHandle}`}
            />
          )}
          <form method="POST" action={action}>
            <input type="hidden" name="flow" value={flowHandle} />
            {canFinish && (
              <button
                type="submit"
                className="h-[38px] rounded-md bg-primary px-4 text-sm font-semibold text-primary-foreground hover:bg-primary/90"
              >
                Finish connecting
              </button>
            )}
            {canCancel && (
              <button
                type="submit"
                name="decision"
                value="deny"
                className="ml-2 h-[38px] rounded-md border px-4 text-sm font-semibold text-foreground hover:bg-accent/40"
              >
                Cancel
              </button>
            )}
            {loopbackClient && (
              <label className="mt-2 flex items-center gap-2 text-[13px] text-muted-foreground">
                <input type="checkbox" name="delivery" value="manual" />
                My client is on a remote or SSH machine
              </label>
            )}
          </form>
        </div>
      </div>
    </div>
  );
};

export default ConnectFlowBanner;
