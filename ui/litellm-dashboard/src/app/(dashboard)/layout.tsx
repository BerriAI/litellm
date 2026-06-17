"use client";

import React, { Suspense, useState } from "react";
import Navbar from "@/components/navbar";
import LoadingScreen from "@/components/common_components/LoadingScreen";
import { ThemeProvider } from "@/contexts/ThemeContext";
import { useAuth } from "@/contexts/AuthContext";
import SidebarProvider from "@/app/(dashboard)/components/SidebarProvider";
import { useRouter, useSearchParams, usePathname } from "next/navigation";
import { DebugWarningBanner } from "@/components/DebugWarningBanner";
import { MIGRATED_PAGES, migratedHref, legacyPageHref, legacyKeyForPathname } from "@/utils/migratedPages";
import { PluginModeProvider, usePluginMode } from "@/contexts/PluginModeContext";

function AgentControlPlaneView() {
  const { agentPlatformUrl, agentPlatformPath } = usePluginMode();
  const { accessToken } = useAuth();

  if (!agentPlatformUrl) {
    return (
      <div className="flex flex-1 items-center justify-center text-gray-500">
        <div className="text-center">
          <p className="text-lg font-medium mb-2">Agent Control Plane</p>
          <p className="text-sm">Configure agent platform URL in settings</p>
        </div>
      </div>
    );
  }

  // Pass the user's litellm virtual key — LAP validates it against litellm's /key/info.
  // This propagates litellm's user hierarchy (role, team, budget) into the agent control plane.
  const params = accessToken ? `?token=${encodeURIComponent(accessToken)}` : "";
  const iframeSrc = `${agentPlatformUrl}${agentPlatformPath}${params}`;

  return (
    <iframe
      src={iframeSrc}
      style={{
        width: "100%",
        height: "100%",
        border: "none",
        flex: 1,
        minHeight: "calc(100vh - 56px)",
      }}
      title="Agent Control Plane"
      allow="clipboard-read; clipboard-write"
    />
  );
}

function DashboardShell({ children }: { children: React.ReactNode }) {
  const router = useRouter();
  const searchParams = useSearchParams();
  const pathname = usePathname();
  const { accessToken } = useAuth();
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const { mode } = usePluginMode();

  const page = legacyKeyForPathname(pathname) || searchParams.get("page") || "api-keys";

  const navigateToPage = (newPage: string) => {
    const migratedRoute = MIGRATED_PAGES[newPage];
    router.push(migratedRoute ? migratedHref(migratedRoute) : legacyPageHref(newPage));
  };

  return (
    <div className="flex flex-col min-h-screen">
      <Navbar
        accessToken={accessToken}
        isPublicPage={false}
        sidebarCollapsed={sidebarCollapsed}
        onToggleSidebar={() => setSidebarCollapsed((v) => !v)}
      />
      <DebugWarningBanner accessToken={accessToken} />
      <div className="flex flex-1">
        <div className="mt-2">
          <SidebarProvider setPage={navigateToPage} defaultSelectedKey={page} sidebarCollapsed={sidebarCollapsed} />
        </div>
        {mode === "litellm-platform-plugin" ? (
          <div className="flex-1 flex">
            <AgentControlPlaneView />
          </div>
        ) : (
          <main className="flex-1">{children}</main>
        )}
      </div>
    </div>
  );
}

function LayoutContent({ children }: { children: React.ReactNode }) {
  const searchParams = useSearchParams();
  const { accessToken, authLoading } = useAuth();
  const isInvitationFlow = Boolean(searchParams.get("invitation_id"));

  if (authLoading) {
    return <LoadingScreen />;
  }

  return (
    <ThemeProvider accessToken={accessToken}>
      {isInvitationFlow ? children : <DashboardShell>{children}</DashboardShell>}
    </ThemeProvider>
  );
}

export default function Layout({ children }: { children: React.ReactNode }) {
  return (
    <Suspense fallback={<LoadingScreen />}>
      <PluginModeProvider>
        <LayoutContent>{children}</LayoutContent>
      </PluginModeProvider>
    </Suspense>
  );
}
