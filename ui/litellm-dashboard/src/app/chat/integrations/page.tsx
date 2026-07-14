"use client";

import { Suspense, useEffect } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { useChatShell } from "@/contexts/ChatShellContext";
import MCPAppsPanel from "@/components/chat/MCPAppsPanel";
import ConnectFlowBanner from "@/components/chat/ConnectFlowBanner";

// useSearchParams() requires a Suspense boundary for static export.
function IntegrationsPageContent() {
  const { accessToken, selectedMCPServers, setSelectedMCPServers } = useChatShell();
  const router = useRouter();
  const searchParams = useSearchParams();
  const oauthReturn = searchParams.get("mcpOauthReturn");
  // Set by the gateway DCR authorize when a DCR client sends the user here to
  // authorize servers before finishing sign-in (see gateway_dcr_flow.py). The
  // handle keys the sealed per-flow cookie; connect_client is the client origin
  // for display only. connect_flow is NOT cleaned from the URL: the finish form
  // needs it, and the sealed cookie (not the URL) is the security boundary.
  const connectFlow = searchParams.get("connect_flow");
  const connectClient = searchParams.get("connect_client");

  // Clean up the OAuth return param after it's been consumed — real routing means
  // we no longer need it to pick a tab, but it should not linger in the address bar.
  useEffect(() => {
    if (oauthReturn) {
      const url = new URL(window.location.href);
      url.searchParams.delete("mcpOauthReturn");
      router.replace(url.pathname + url.search);
    }
  }, [oauthReturn, router]);

  return (
    <div className="flex-1 min-h-0 overflow-auto w-full py-8 px-8">
      {connectFlow && <ConnectFlowBanner flowHandle={connectFlow} clientOrigin={connectClient} />}
      <MCPAppsPanel accessToken={accessToken} selectedServers={selectedMCPServers} onChange={setSelectedMCPServers} />
    </div>
  );
}

export default function IntegrationsPage() {
  return (
    <Suspense>
      <IntegrationsPageContent />
    </Suspense>
  );
}
