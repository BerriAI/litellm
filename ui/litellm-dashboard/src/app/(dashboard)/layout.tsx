"use client";

import React from "react";
import Navbar from "@/components/navbar";
import { ThemeProvider } from "@/contexts/ThemeContext";
import Sidebar2 from "@/app/(dashboard)/components/Sidebar2";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

export default function Layout({ children }: { children: React.ReactNode }) {
  const { accessToken, userRole } = useAuthorized();
  const [sidebarCollapsed, setSidebarCollapsed] = React.useState(false);

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
            <Sidebar2
              collapsed={sidebarCollapsed}
              accessToken={accessToken}
              userRole={userRole}
              defaultSelectedKey={""}
            />
          </div>
          <main className="flex-1">{children}</main>
        </div>
      </div>
    </ThemeProvider>
  );
}
