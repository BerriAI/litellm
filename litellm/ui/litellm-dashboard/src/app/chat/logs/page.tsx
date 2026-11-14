"use client";

import { useChatShell } from "@/contexts/ChatShellContext";
import LogsPanel from "@/components/chat/LogsPanel";

export default function LogsPage() {
  const { accessToken, userId } = useChatShell();

  return (
    <div className="flex-1 min-h-0 overflow-auto w-full py-8 px-8">
      <LogsPanel accessToken={accessToken} userId={userId} />
    </div>
  );
}
