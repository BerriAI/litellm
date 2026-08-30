"use client";

import { Suspense, useEffect, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import MCPAppsPanel from "@/components/chat/MCPAppsPanel";
import ConnectFlowBanner from "@/components/chat/ConnectFlowBanner";

function ConnectPageContent() {
  const { accessToken } = useAuthorized();
  const [selectedServers, setSelectedServers] = useState<string[]>([]);
  const router = useRouter();
  const searchParams = useSearchParams();
  const oauthReturn = searchParams.get("mcpOauthReturn");
  const connectFlow = searchParams.get("connect_flow");
  const connectClient = searchParams.get("connect_client");

  useEffect(() => {
    if (oauthReturn) {
      const url = new URL(window.location.href);
      url.searchParams.delete("mcpOauthReturn");
      router.replace(url.pathname + url.search);
    }
  }, [oauthReturn, router]);

  return (
    <div className="mx-auto w-full max-w-5xl px-8 py-8">
      {connectFlow && <ConnectFlowBanner flowHandle={connectFlow} clientOrigin={connectClient} />}
      <MCPAppsPanel
        accessToken={accessToken ?? ""}
        selectedServers={selectedServers}
        onChange={setSelectedServers}
        connectMode={!!connectFlow}
      />
    </div>
  );
}

export default function ConnectPage() {
  return (
    <Suspense>
      <ConnectPageContent />
    </Suspense>
  );
}
