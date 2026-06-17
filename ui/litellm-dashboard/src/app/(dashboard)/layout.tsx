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

function DashboardShell({ children }: { children: React.ReactNode }) {
  const router = useRouter();
  const searchParams = useSearchParams();
  const pathname = usePathname();
  const { accessToken } = useAuth();
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

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
        <main className="flex-1">{children}</main>
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
      <LayoutContent>{children}</LayoutContent>
    </Suspense>
  );
}
