"use client";

import React, { useEffect, useState } from "react";
import Navbar from "@/components/navbar";
import { ThemeProvider } from "@/contexts/ThemeContext";
import Sidebar2 from "@/app/(dashboard)/components/Sidebar2";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { useRouter, useSearchParams } from "next/navigation";

export default function Layout({ children }: { children: React.ReactNode }) {
  const router = useRouter();
  const searchParams = useSearchParams();
  const { accessToken, userRole } = useAuthorized();
  const [sidebarCollapsed, setSidebarCollapsed] = React.useState(false);
  const [page, setPage] = useState(() => {
    return searchParams.get("page") || "api-keys";
  });

  const updatePage = (newPage: string) => {
    const newSearchParams = new URLSearchParams(searchParams);
    newSearchParams.set("page", newPage);
    router.push(`/?${newSearchParams.toString()}`); // absolute, not relative
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
          userID={null}
          userEmail={null}
          userRole={null}
          premiumUser={false}
          proxySettings={undefined}
          setProxySettings={function (value: any): void {
            throw new Error("Function not implemented.");
          }}
          accessToken={null}
        />
        <div className="flex flex-1 overflow-auto">
          <div className="mt-2">
            <Sidebar2 defaultSelectedKey={page} setPage={updatePage} accessToken={accessToken} userRole={userRole} />
          </div>
          <main className="flex-1">{children}</main>
        </div>
      </div>
    </ThemeProvider>
  );
}
