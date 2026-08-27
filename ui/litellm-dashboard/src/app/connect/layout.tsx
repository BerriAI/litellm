"use client";

import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import Navbar from "@/components/navbar";
import { ThemeProvider } from "@/contexts/ThemeContext";

export default function ConnectLayout({ children }: { children: React.ReactNode }) {
  const { accessToken, isAuthorized, isLoading } = useAuthorized();

  if (isLoading || !isAuthorized) return null;

  return (
    <ThemeProvider accessToken={accessToken}>
      <div className="flex h-screen flex-col">
        <Navbar accessToken={accessToken} isPublicPage={false} />
        <div className="min-h-0 flex-1 overflow-auto">{children}</div>
      </div>
    </ThemeProvider>
  );
}
