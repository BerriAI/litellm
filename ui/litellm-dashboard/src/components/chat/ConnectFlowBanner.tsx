"use client";

import { useLanguage } from "@/contexts/LanguageContext";
import React from "react";
import { CheckCircle } from "lucide-react";
import { getProxyBaseUrl, ConnectFlowStatus } from "@/components/networking";
import { OAuth2ConnectButton } from "@/components/chat/MCPAppsPanel";
import type { Translate } from "@/i18n/language";

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

// 仅翻译说明文案，客户端来源和服务器名称保留原值。
const copyFor = (flow: ConnectFlowStatus | undefined, failed: boolean, t: Translate): readonly [string, string] => {
  const clientLabel = flow?.client_origin ?? t("the application");
  const serverLabel = flow?.server_name ?? t("the requested MCP server");
  const serverNames = flow?.servers?.map((server) => server.server_name).join(", ") ?? "";
  if (failed || flow === undefined || flow.state === "stale") {
    return [
      t("The connection cannot continue"),
      t("The gateway could not validate this connection. Cancel to return to {client}.", { client: clientLabel }),
    ];
  }
  if (flow.state === "unscoped") {
    return [
      t("Connect your MCP servers to {client}", { client: clientLabel }),
      t("Authorize the servers you want to use below, then click Finish connecting to return to {client}.", {
        client: clientLabel,
      }),
    ];
  }
  if (flow.state === "multi") {
    const allServersConnected =
      flow.servers !== null && flow.servers.length > 0 && flow.servers.every((server) => server.connected);
    return [
      t("Allow {client} to use {servers}", { client: clientLabel, servers: serverNames }),
      allServersConnected
        ? t("Click Finish connecting to give {client} access to {servers} as you.", {
            client: clientLabel,
            servers: serverNames,
          })
        : t("Authorize each requested server below to continue, or cancel to send {client} away.", {
            client: clientLabel,
          }),
    ];
  }
  if (flow.state === "interactive" && !flow.connected) {
    return [
      t("Allow {client} to use {servers}", { client: clientLabel, servers: serverLabel }),
      t("Authorize {server} below to continue, or cancel to send {client} away.", {
        server: serverLabel,
        client: clientLabel,
      }),
    ];
  }
  return [
    t("Allow {client} to use {servers}", { client: clientLabel, servers: serverLabel }),
    t("Click Finish connecting to give {client} access to {servers} as you.", {
      client: clientLabel,
      servers: serverLabel,
    }),
  ];
};

const canFinishConnectFlow = (state: ConnectFlowStatus["state"], flow: ConnectFlowStatus | undefined): boolean => {
  if (state === "unscoped") return true;
  if (state === "stale" || flow === undefined) return false;
  if (state !== "multi") return flow.connected === true;
  return flow.servers !== null && flow.servers.length > 0 && flow.servers.every((server) => server.connected);
};

const vendorServersFor = (
  state: ConnectFlowStatus["state"],
  flow: ConnectFlowStatus | undefined,
): { server_id: string; server_name: string | null }[] => {
  if (state === "multi") {
    return (flow?.servers ?? [])
      .filter((server) => !server.connected)
      .map(({ server_id, server_name }) => ({ server_id, server_name }));
  }
  if (state !== "interactive" || flow?.connected !== false || flow.server_id === null) return [];
  return [{ server_id: flow.server_id, server_name: flow.server_name }];
};

const ConnectFlowBanner: React.FC<Props> = ({ flowHandle, flow, accessToken, onConnected, failed }) => {
  const { t } = useLanguage();
  const action = `${getProxyBaseUrl()}/authorize/complete`;
  const state = failed || flow === undefined ? "stale" : flow.state;
  const canFinish = canFinishConnectFlow(state, flow);
  const canCancel = state !== "unscoped";
  const loopbackClient = isLoopbackOrigin(flow?.client_origin ?? null);
  const vendorServers = vendorServersFor(state, flow);
  const copy = copyFor(flow, failed, t);

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
          {vendorServers.map((server) => (
            <OAuth2ConnectButton
              key={server.server_id}
              server={server}
              accessToken={accessToken}
              onConnect={onConnected}
              scopes={flow?.requested_scopes}
              variant="button"
              buttonLabel={
                state === "multi"
                  ? t("Connect {server}", { server: server.server_name ?? server.server_id })
                  : undefined
              }
              {...(state === "multi"
                ? {}
                : { autoStartKey: `litellm-mcp-autostart:${flowHandle}:${server.server_id}` })}
            />
          ))}
          <form method="POST" action={action}>
            <input type="hidden" name="flow" value={flowHandle} />
            {canFinish && (
              <button
                type="submit"
                className="h-[38px] rounded-md bg-primary px-4 text-sm font-semibold text-primary-foreground hover:bg-primary/90"
              >
                {t("Finish connecting")}
              </button>
            )}
            {canCancel && (
              <button
                type="submit"
                name="decision"
                value="deny"
                className="ml-2 h-[38px] rounded-md border px-4 text-sm font-semibold text-foreground hover:bg-accent/40"
              >
                {t("Cancel")}
              </button>
            )}
            {loopbackClient && (
              <label className="mt-2 flex items-center gap-2 text-[13px] text-muted-foreground">
                <input type="checkbox" name="delivery" value="manual" />
                {t("My client is on a remote or SSH machine")}
              </label>
            )}
          </form>
        </div>
      </div>
    </div>
  );
};

export default ConnectFlowBanner;
