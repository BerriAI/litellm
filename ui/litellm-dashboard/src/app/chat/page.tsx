"use client";

import { Suspense, useEffect } from "react";
import { useRouter } from "next/navigation";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { useUIConfig } from "@/app/(dashboard)/hooks/uiConfig/useUIConfig";
import { useUISettings } from "@/app/(dashboard)/hooks/uiSettings/useUISettings";
import ChatPage from "@/components/chat/ChatPage";

// ChatPage uses useSearchParams() which requires a Suspense boundary for static export.
const ChatPageContent = () => {
  const { accessToken, userRole, userId, userEmail } = useAuthorized();
  const { data: uiSettings, isLoading: isUISettingsLoading } = useUISettings();
  const { data: uiConfig } = useUIConfig();
  const router = useRouter();

  const uiRoot =
    uiConfig?.server_root_path && uiConfig.server_root_path !== "/"
      ? uiConfig.server_root_path.replace(/\/+$/, "")
      : "";
  const chatEnabled = Boolean(uiSettings?.values?.enable_chat_ui);
  const blocked = !isUISettingsLoading && !chatEnabled;

  useEffect(() => {
    if (blocked) router.replace(`${uiRoot}/ui/`);
  }, [blocked, uiRoot, router]);

  if (isUISettingsLoading || blocked) return null;

  return (
    <ChatPage
      accessToken={accessToken ?? ""}
      userRole={userRole ?? ""}
      userId={userId ?? ""}
      userEmail={userEmail ?? ""}
    />
  );
};

const ChatPageRoute = () => (
  <Suspense>
    <ChatPageContent />
  </Suspense>
);

export default ChatPageRoute;
