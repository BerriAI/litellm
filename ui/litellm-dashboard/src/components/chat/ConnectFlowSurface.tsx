"use client";

import React, { useEffect, useState } from "react";
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

/** Renders the sealed gateway connect flow without trusting URL context. */
const ConnectFlowSurface: React.FC<Props> = ({ accessToken, selectedServers, onChange }) => {
  const router = useRouter();
  const searchParams = useSearchParams();
  const oauthReturn = searchParams.get("mcpOauthReturn");
  const connectFlow = searchParams.get("connect_flow");
  const [flowSelection, setFlowSelection] = useState<{ handle: string | null; serverIds: string[] }>({
    handle: null,
    serverIds: [],
  });
  const selectedServerIds = flowSelection.handle === connectFlow ? flowSelection.serverIds : [];

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

  if (connectFlow === null) {
    return <MCPAppsPanel accessToken={accessToken} selectedServers={selectedServers} onChange={onChange} />;
  }

  return (
    <>
      <ConnectFlowBanner
        flowHandle={connectFlow}
        flow={flow}
        accessToken={accessToken}
        onConnected={refetch}
        failed={isError}
        selectedServers={selectedServerIds}
      />
      {flow?.state === "unscoped" && (
        <MCPAppsPanel
          accessToken={accessToken}
          selectedServers={selectedServerIds}
          onChange={(serverIds) => setFlowSelection({ handle: connectFlow, serverIds })}
          connectMode
        />
      )}
    </>
  );
};

export default ConnectFlowSurface;
