"use client";

import React, { useEffect } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { useQuery } from "@tanstack/react-query";
import MCPAppsPanel from "@/components/chat/MCPAppsPanel";
import ConnectFlowBanner from "@/components/chat/ConnectFlowBanner";
import { fetchConnectFlow } from "@/components/networking";

interface Props {
  accessToken: string;
  selectedServers: string[];
  onChange: (servers: string[]) => void;
}

/**
 * The one place the gateway DCR connect flow is turned into a page.
 *
 * The URL carries only connect_flow, the handle keying the sealed HttpOnly cookie that
 * gateway_dcr_flow.py's authorize set. Everything shown or acted on comes from asking the
 * gateway about that handle: the client origin, the one server a resource-scoped flow named,
 * and whether its vendor OAuth is done. So a crafted link can neither relabel the consent
 * screen nor point the vendor round trip at a server the flow never named, and the finish
 * step refuses a scoped mint until the gateway itself sees the credential.
 *
 * connect_flow is NOT cleaned from the URL: the finish form needs it. mcpOauthReturn is
 * cleaned once consumed; whether a vendor trip already happened is remembered per flow in
 * sessionStorage by the panel, so browser Back cannot restart one.
 */
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
  const { data: flow, error, refetch } = useQuery(flowQuery);

  const scopedServerId = flow?.server_id ?? null;
  const scopedConnected = scopedServerId !== null && flow?.connected === true;
  const scopedFlow = flow?.state === "interactive" || flow?.state === "m2m";
  const flowLoading = connectFlow !== null && flow === undefined && !error;
  const flowUnavailable = connectFlow !== null && (flowLoading || error !== null);
  const staleFlow = flow?.state === "stale";
  const flowServerLabel = staleFlow || flowUnavailable ? "the requested MCP server" : flow?.server_name;
  const bannerClientOrigin = flow?.client_origin ?? null;

  return (
    <>
      {connectFlow !== null && (
        <ConnectFlowBanner
          flowHandle={connectFlow}
          clientOrigin={bannerClientOrigin}
          serverLabel={flowServerLabel}
          serverUnavailable={flowUnavailable || staleFlow}
          canFinish={flow !== undefined && (flow.state === "unscoped" || flow.state === "m2m" || scopedConnected)}
          canCancel
        />
      )}
      {(connectFlow === null || (flow !== undefined && !staleFlow)) && (
        <MCPAppsPanel
          accessToken={accessToken}
          selectedServers={selectedServers}
          onChange={onChange}
          connectMode={flow !== undefined}
          scopedServerId={scopedFlow ? scopedServerId : null}
          autoStartKey={
            flow?.state === "interactive" && !scopedConnected ? `litellm-mcp-autostart:${connectFlow}` : null
          }
          onScopedConnected={refetch}
        />
      )}
    </>
  );
};

export default ConnectFlowSurface;
