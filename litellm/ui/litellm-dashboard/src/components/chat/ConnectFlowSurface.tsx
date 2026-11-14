"use client";

import React, { useEffect, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { useQuery } from "@tanstack/react-query";
import MCPAppsPanel, { type ServerListReport } from "@/components/chat/MCPAppsPanel";
import ConnectFlowBanner, { type FinishSelection } from "@/components/chat/ConnectFlowBanner";
import { fetchConnectFlow } from "@/components/networking";
import { pendingUserMcpOAuthServerId } from "@/hooks/useUserMcpOAuthFlow";

interface Props {
  accessToken: string;
  selectedServers: string[];
  onChange: (servers: string[]) => void;
}

const offServersKey = (flow: string) => `litellm-connect-off:${flow}`;

const storedServerIds = (key: string): string[] => {
  try {
    const value: unknown = JSON.parse(sessionStorage.getItem(key) ?? "null");
    return Array.isArray(value) ? value.filter((item): item is string => typeof item === "string") : [];
  } catch {
    return [];
  }
};

const rememberServerIds = (key: string, serverIds: string[]) => {
  try {
    sessionStorage.setItem(key, JSON.stringify(serverIds));
  } catch {}
};

const finishSelectionFor = (connected: ServerListReport, offServerIds: readonly string[]): FinishSelection => {
  if (connected === "saving-login") return { status: "waiting" };
  if (typeof connected === "string") {
    return offServerIds.length === 0 ? { status: "ready", serverIds: [] } : { status: "waiting" };
  }
  if (connected.on.length === connected.all.length) return { status: "ready", serverIds: [] };
  if (connected.on.length === 0) return { status: "nothing-on" };
  return { status: "ready", serverIds: connected.on };
};

/** Renders the sealed gateway connect flow without trusting URL context. */
const ConnectFlowSurface: React.FC<Props> = ({ accessToken, selectedServers, onChange }) => {
  const router = useRouter();
  const searchParams = useSearchParams();
  const oauthReturn = searchParams.get("mcpOauthReturn");
  const connectFlow = searchParams.get("connect_flow");

  useEffect(() => {
    if (oauthReturn) {
      const url = new URL(window.location.href);
      url.searchParams.delete("mcpOauthReturn");
      router.replace(url.pathname + url.search);
    }
  }, [oauthReturn, router]);

  const flowQuery = {
    queryKey: ["gateway-connect-flow", connectFlow],
    queryFn: () => fetchConnectFlow(connectFlow!),
    enabled: !!connectFlow,
    retry: false,
  };
  const { data: flow, isError, refetch } = useQuery(flowQuery);
  const [connected, setConnected] = useState<ServerListReport>(() =>
    pendingUserMcpOAuthServerId() === null ? "loading" : "saving-login",
  );
  const [offServerIds, setOffServerIds] = useState(() =>
    connectFlow ? storedServerIds(offServersKey(connectFlow)) : [],
  );

  const changeOffServerIds = (serverIds: string[]) => {
    if (connectFlow) rememberServerIds(offServersKey(connectFlow), serverIds);
    setOffServerIds(serverIds);
  };

  if (connectFlow === null) {
    return <MCPAppsPanel accessToken={accessToken} selectedServers={selectedServers} onChange={onChange} />;
  }

  const narrowable = flow?.state === "unscoped" && flow.selectable !== false;

  return (
    <>
      <ConnectFlowBanner
        flowHandle={connectFlow}
        flow={flow}
        accessToken={accessToken}
        onConnected={refetch}
        failed={isError}
        finish={narrowable ? finishSelectionFor(connected, offServerIds) : undefined}
      />
      {flow?.state === "unscoped" && (
        <MCPAppsPanel
          accessToken={accessToken}
          selectedServers={selectedServers}
          onChange={onChange}
          connectMode
          offServerIds={offServerIds}
          onOffServerIdsChange={changeOffServerIds}
          lockedOn={narrowable ? flow.required_server_ids : "all"}
          onConnectedServers={setConnected}
        />
      )}
    </>
  );
};

export default ConnectFlowSurface;
