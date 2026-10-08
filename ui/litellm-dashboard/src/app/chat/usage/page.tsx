"use client";

import { useChatShell } from "@/contexts/ChatShellContext";
import UsagePanel from "@/components/chat/UsagePanel";

export default function UsagePage() {
  const { accessToken, userId } = useChatShell();

  return (
    <div className="flex-1 min-h-0 overflow-auto w-full py-8 px-8">
      <UsagePanel accessToken={accessToken} userId={userId} />
    </div>
  );
}
