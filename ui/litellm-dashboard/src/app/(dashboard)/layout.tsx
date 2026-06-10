"use client";

import React, { Suspense, useEffect, useState } from "react";
import Navbar from "@/components/navbar";
import { ThemeProvider } from "@/contexts/ThemeContext";
import SidebarProvider from "@/app/(dashboard)/components/SidebarProvider";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { useRouter, useSearchParams, usePathname } from "next/navigation";
import { DebugWarningBanner } from "@/components/DebugWarningBanner";
import { MIGRATED_PAGES, migratedHref, legacyPageHref, legacyKeyForPathname } from "@/utils/migratedPages";

function LayoutContent({ children }: { children: React.ReactNode }) {
  const router = useRouter();
  const searchParams = useSearchParams();
  const pathname = usePathname();
  const { accessToken } = useAuthorized();
  const [sidebarCollapsed, setSidebarCollapsed] = React.useState(false);
  const [page, setPage] = useState(() => {
    return legacyKeyForPathname(pathname) || searchParams.get("page") || "api-keys";
  });

  const handleSetPage = (newPage: string) => {
    const migratedRoute = MIGRATED_PAGES[newPage];
    router.push(migratedRoute ? migratedHref(migratedRoute) : legacyPageHref(newPage));
    setPage(newPage);
  };

  useEffect(() => {
    setPage(legacyKeyForPathname(pathname) || searchParams.get("page") || "api-keys");
  }, [pathname, searchParams]);

  const toggleSidebar = () => setSidebarCollapsed((v) => !v);

  return (
    <ThemeProvider accessToken={""}>
      <div className="flex flex-col min-h-screen">
        <Navbar
          isPublicPage={false}
          sidebarCollapsed={sidebarCollapsed}
          onToggleSidebar={toggleSidebar}
          proxySettings={undefined}
          setProxySettings={() => {}}
          accessToken={accessToken}
        />
        <DebugWarningBanner accessToken={accessToken} />
        <div className="flex flex-1 overflow-auto">
          <div className="mt-2">
            <SidebarProvider setPage={handleSetPage} defaultSelectedKey={page} sidebarCollapsed={sidebarCollapsed} />
          </div>
          <main className="flex-1">{children}</main>
        </div>
      </div>
    </ThemeProvider>
  );
}

export default function Layout({ children }: { children: React.ReactNode }) {
  return (
    <Suspense fallback={<div className="flex items-center justify-center min-h-screen">Loading...</div>}>
      <LayoutContent>{children}</LayoutContent>
    </Suspense>
  );
}
