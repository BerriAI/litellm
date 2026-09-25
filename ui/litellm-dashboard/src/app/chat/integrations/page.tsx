"use client";

import { Suspense } from "react";
import { useChatShell } from "@/contexts/ChatShellContext";
import ConnectFlowSurface from "@/components/chat/ConnectFlowSurface";

// useSearchParams() requires a Suspense boundary for static export.
function IntegrationsPageContent() {
  const { accessToken, selectedMCPServers, setSelectedMCPServers } = useChatShell();

  return (
    <div className="flex-1 min-h-0 overflow-auto w-full py-8 px-8">
      <ConnectFlowSurface
        accessToken={accessToken}
        selectedServers={selectedMCPServers}
        onChange={setSelectedMCPServers}
      />
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
