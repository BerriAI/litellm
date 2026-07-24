"use client";

import { Suspense, useEffect, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import MCPAppsPanel from "@/components/chat/MCPAppsPanel";

function ConnectPageContent() {
  const { accessToken } = useAuthorized();
  const [selectedServers, setSelectedServers] = useState<string[]>([]);
  const router = useRouter();
  const searchParams = useSearchParams();
  const oauthReturn = searchParams.get("mcpOauthReturn");

  useEffect(() => {
    if (oauthReturn) {
      const url = new URL(window.location.href);
      url.searchParams.delete("mcpOauthReturn");
      router.replace(url.pathname + url.search);
    }
  }, [oauthReturn, router]);

  return (
    <div className="mx-auto w-full max-w-5xl px-8 py-8">
      <MCPAppsPanel accessToken={accessToken ?? ""} selectedServers={selectedServers} onChange={setSelectedServers} />
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
