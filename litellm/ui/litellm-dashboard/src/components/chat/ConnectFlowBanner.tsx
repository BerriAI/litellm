"use client";

import React from "react";
import { CheckCircle } from "lucide-react";
import type { TFunction } from "i18next";
import { useTranslation } from "react-i18next";
import { getProxyBaseUrl, ConnectFlowStatus } from "@/components/networking";
import { OAuth2ConnectButton } from "@/components/chat/MCPAppsPanel";
import { Button } from "@/components/ui/button";
import i18n from "@/i18n/i18n";

interface Props {
  flowHandle: string;
  flow?: ConnectFlowStatus;
  accessToken: string;
  onConnected: () => void;
  failed: boolean;
  finish?: FinishSelection;
}

export type FinishSelection =
  | { status: "waiting" }
  | { status: "nothing-on" }
  | { status: "ready"; serverIds: readonly string[] };

const FINISH_WITHOUT_SERVERS: FinishSelection = { status: "ready", serverIds: [] };

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

const copyFor = (
  flow: ConnectFlowStatus | undefined,
  failed: boolean,
  t: TFunction = i18n.t.bind(i18n),
): readonly [string, string] => {
  const clientLabel = flow?.client_origin ?? t("chat.connectFlowBanner.defaultClient");
  const serverLabel = flow?.server_name ?? t("chat.connectFlowBanner.defaultServer");
  const serverNames = flow?.servers?.map((server) => server.server_name).join(", ") ?? "";
  if (failed || flow === undefined || flow.state === "stale") {
    return [
      t("chat.connectFlowBanner.stale.title"),
      t("chat.connectFlowBanner.stale.description", { client: clientLabel }),
    ];
  }
  if (flow.state === "unscoped") {
    return [
      t("chat.connectFlowBanner.unscoped.title", { client: clientLabel }),
      flow.selectable === false
        ? t("chat.connectFlowBanner.unscoped.fixed", { client: clientLabel })
        : t("chat.connectFlowBanner.unscoped.description", { client: clientLabel }),
    ];
  }
  if (flow.state === "multi") {
    const allServersConnected =
      flow.servers !== null && flow.servers.length > 0 && flow.servers.every((server) => server.connected);
    return [
      t("chat.connectFlowBanner.multi.title", { client: clientLabel, servers: serverNames }),
      allServersConnected
        ? t("chat.connectFlowBanner.multi.ready", { client: clientLabel, servers: serverNames })
        : t("chat.connectFlowBanner.multi.pending", { client: clientLabel }),
    ];
  }
  if (flow.state === "interactive" && !flow.connected) {
    return [
      t("chat.connectFlowBanner.single.title", { client: clientLabel, server: serverLabel }),
      t("chat.connectFlowBanner.single.pending", { client: clientLabel, server: serverLabel }),
    ];
  }
  return [
    t("chat.connectFlowBanner.single.title", { client: clientLabel, server: serverLabel }),
    t("chat.connectFlowBanner.single.ready", { client: clientLabel, server: serverLabel }),
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

const ConnectFlowBanner: React.FC<Props> = ({
  flowHandle,
  flow,
  accessToken,
  onConnected,
  failed,
  finish = FINISH_WITHOUT_SERVERS,
}) => {
  const { t } = useTranslation();
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
                  ? t("chat.connectFlowBanner.connectServer", { server: server.server_name ?? server.server_id })
                  : undefined
              }
              {...(state === "multi"
                ? {}
                : { autoStartKey: `litellm-mcp-autostart:${flowHandle}:${server.server_id}` })}
            />
          ))}
          <form method="POST" action={action}>
            <input type="hidden" name="flow" value={flowHandle} />
            {finish.status === "ready" &&
              finish.serverIds.map((serverId) => <input key={serverId} type="hidden" name="server" value={serverId} />)}
            {canFinish && (
              <Button
                type="submit"
                disabled={finish.status !== "ready"}
                aria-describedby={finish.status === "nothing-on" ? "connect-flow-nothing-on" : undefined}
                className="h-[38px] px-4 font-semibold"
              >
                {t("chat.connectFlowBanner.finish")}
              </Button>
            )}
            {canCancel && (
              <button
                type="submit"
                name="decision"
                value="deny"
                className="ml-2 h-[38px] rounded-md border px-4 text-sm font-semibold text-foreground hover:bg-accent/40"
              >
                {t("chat.connectFlowBanner.cancel")}
              </button>
            )}
            {loopbackClient && (
              <label className="mt-2 flex items-center gap-2 text-[13px] text-muted-foreground">
                <input type="checkbox" name="delivery" value="manual" />
                {t("chat.connectFlowBanner.manualDelivery")}
              </label>
            )}
            {finish.status === "nothing-on" && (
              <p id="connect-flow-nothing-on" className="mt-2 text-[13px] text-muted-foreground">
                {t("chat.connectFlowBanner.nothingOn")}
              </p>
            )}
          </form>
        </div>
      </div>
    </div>
  );
};

export default ConnectFlowBanner;
