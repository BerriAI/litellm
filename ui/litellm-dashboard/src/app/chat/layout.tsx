"use client";

import { Suspense, useEffect } from "react";
import { useRouter } from "next/navigation";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { useUISettings } from "@/app/(dashboard)/hooks/uiSettings/useUISettings";
import Navbar from "@/components/navbar";
import { ThemeProvider } from "@/contexts/ThemeContext";
import { ChatShellProvider } from "@/contexts/ChatShellContext";
import ChatShell from "@/components/chat/ChatShell";
import { migratedHref } from "@/utils/migratedPages";

// ChatShellProvider uses useSearchParams(), which requires a Suspense boundary for static export.
function ChatLayoutContent({ children }: { children: React.ReactNode }) {
  const { accessToken, userRole, userId, userEmail, premiumUser } = useAuthorized();
  const { data: uiSettings, isLoading: isUISettingsLoading } = useUISettings();
  const router = useRouter();

  const chatEnabled = Boolean(uiSettings?.values?.enable_chat_ui);
  const blocked = !isUISettingsLoading && !chatEnabled;

  useEffect(() => {
    if (blocked) router.replace(migratedHref(""));
  }, [blocked, router]);

  if (isUISettingsLoading || blocked) return null;

  return (
    <ThemeProvider accessToken={accessToken}>
      <div className="flex h-screen flex-col">
        <Navbar accessToken={accessToken} isPublicPage={false} />
        <div className="min-h-0 flex-1">
          <ChatShellProvider
            accessToken={accessToken ?? ""}
            userId={userId ?? ""}
            userEmail={userEmail ?? ""}
            userRole={userRole ?? ""}
            premiumUser={premiumUser ?? false}
          >
            <ChatShell>{children}</ChatShell>
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
