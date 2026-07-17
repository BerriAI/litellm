"use client";

import { useChatShell } from "@/contexts/ChatShellContext";
import KeysPanel from "@/components/chat/KeysPanel";

export default function ApiKeysPage() {
  const { accessToken, userId, premiumUser } = useChatShell();

  return (
    <div className="flex-1 min-h-0 overflow-auto w-full py-8 px-8">
      <KeysPanel accessToken={accessToken} userId={userId} premiumUser={premiumUser} />
    </div>
  );
}
