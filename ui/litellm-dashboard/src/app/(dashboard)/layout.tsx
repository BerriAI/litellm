"use client";

import React, { useEffect, useState } from "react";
import Navbar from "@/components/navbar";
import { ThemeProvider } from "@/contexts/ThemeContext";
import Sidebar2 from "@/app/(dashboard)/components/Sidebar2";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { useRouter, useSearchParams } from "next/navigation";

/** ---- BASE URL HELPERS ---- */
function normalizeBasePrefix(raw: string | undefined | null): string {
  const trimmed = (raw ?? "").trim();
  if (!trimmed) return "";
  const core = trimmed.replace(/^\/+/, "").replace(/\/+$/, "");
  return core ? `/${core}/` : "/";
}
const BASE_PREFIX = normalizeBasePrefix(process.env.NEXT_PUBLIC_BASE_URL);
function withBase(path: string): string {
  const body = path.startsWith("/") ? path.slice(1) : path;
  const combined = `${BASE_PREFIX}${body}`;
  return combined.startsWith("/") ? combined : `/${combined}`;
}
/** -------------------------------- */

export default function Layout({ children }: { children: React.ReactNode }) {
  const router = useRouter();
  const searchParams = useSearchParams();
  const { accessToken, userRole, userId, userEmail, premiumUser } = useAuthorized();
  const [sidebarCollapsed, setSidebarCollapsed] = React.useState(false);
  const [page, setPage] = useState(() => {
    return searchParams.get("page") || "api-keys";
  });

  const updatePage = (newPage: string) => {
    const newSearchParams = new URLSearchParams(searchParams);
    newSearchParams.set("page", newPage);
    router.push(withBase(`/?${newSearchParams.toString()}`)); // always under BASE
    setPage(newPage);
  };

  useEffect(() => {
    setPage(searchParams.get("page") || "api-keys");
  }, [searchParams]);

  const toggleSidebar = () => setSidebarCollapsed((v) => !v);

  return (
    <ThemeProvider accessToken={""}>
      <div className="flex flex-col min-h-screen">
        <Navbar
          isPublicPage={false}
          sidebarCollapsed={sidebarCollapsed}
          onToggleSidebar={toggleSidebar}
          userID={userId}
          userEmail={userEmail}
          userRole={userRole}
          premiumUser={premiumUser}
          proxySettings={undefined}
          setProxySettings={() => {}}
          accessToken={accessToken}
        />
        <div className="flex flex-1 overflow-auto">
          <div className="mt-2">
            <Sidebar2 defaultSelectedKey={page} accessToken={accessToken} userRole={userRole} />
          </div>
          <main className="flex-1">{children}</main>
        </div>
      </div>
    </ThemeProvider>
  );
}
