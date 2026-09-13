"use client";

import { useChatShell } from "@/contexts/ChatShellContext";
import MCPCredentialsTab from "@/components/chat/MCPCredentialsTab";

export default function CredentialsPage() {
  const { accessToken } = useChatShell();

  return (
    <div className="flex-1 min-h-0 overflow-auto w-full py-8 px-8">
      <MCPCredentialsTab accessToken={accessToken} />
    </div>
  );
}
