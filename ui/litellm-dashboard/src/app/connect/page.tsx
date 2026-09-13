"use client";

import { Suspense, useState } from "react";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import ConnectFlowSurface from "@/components/chat/ConnectFlowSurface";

function ConnectPageContent() {
  const { accessToken } = useAuthorized();
  const [selectedServers, setSelectedServers] = useState<string[]>([]);

  return (
    <div className="mx-auto w-full max-w-5xl px-8 py-8">
      <ConnectFlowSurface
        accessToken={accessToken ?? ""}
        selectedServers={selectedServers}
        onChange={setSelectedServers}
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
