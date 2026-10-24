"use client";

import { Suspense, useEffect } from "react";
import { useRouter } from "next/navigation";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { useUISettings } from "@/app/(dashboard)/hooks/uiSettings/useUISettings";
import { ThemeProvider } from "@/contexts/ThemeContext";
import { ChatShellProvider } from "@/contexts/ChatShellContext";
import { PluginModeProvider } from "@/contexts/PluginModeContext";
import ChatShell from "@/components/chat/ChatShell";
import { uiHref } from "@/utils/uiHref";
import "@/components/chat/moyai/moyai-chat.css";

// The nuqs Next adapter uses useSearchParams, so keep the chat tree behind Suspense.
function ChatLayoutContent({ children }: { children: React.ReactNode }) {
  const {
    accessToken,
    userRole,
    userId,
    userEmail,
    premiumUser,
    isLoading: isIdentityLoading,
    isAuthorized,
  } = useAuthorized();
  const { data: uiSettings, isLoading: isUISettingsLoading } = useUISettings();
  const router = useRouter();

  const chatEnabled = Boolean(uiSettings?.values?.enable_chat_ui);
  const blocked = !isUISettingsLoading && !chatEnabled;

  useEffect(() => {
    if (blocked) router.replace(uiHref(""));
  }, [blocked, router]);

  const identityReady = !isIdentityLoading && isAuthorized;
  const chatReady = identityReady && !isUISettingsLoading && chatEnabled;

  if (!chatReady) return null;

  return (
    <ThemeProvider accessToken={accessToken}>
      <link rel="preconnect" href="https://api.fontshare.com" />
      <link rel="preconnect" href="https://cdn.fontshare.com" crossOrigin="anonymous" />
      <link rel="stylesheet" href="https://api.fontshare.com/v2/css?f[]=general-sans@1&display=swap" />
      <div className="flex h-dvh flex-col">
        <div className="min-h-0 flex-1">
          <ChatShellProvider
            accessToken={accessToken ?? ""}
            userId={userId ?? ""}
            userEmail={userEmail ?? ""}
            userRole={userRole ?? ""}
            premiumUser={premiumUser ?? false}
          >
            <PluginModeProvider accessToken={accessToken}>
              <ChatShell>{children}</ChatShell>
            </PluginModeProvider>
          </ChatShellProvider>
        </div>
      </div>
    </ThemeProvider>
  );
}

export default function ChatLayout({ children }: { children: React.ReactNode }) {
  return (
    <Suspense>
      <ChatLayoutContent>{children}</ChatLayoutContent>
    </Suspense>
  );
}
