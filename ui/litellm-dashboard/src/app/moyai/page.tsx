"use client";

import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import Navbar from "@/components/navbar";
import MoyaiLanding from "@/components/moyai/MoyaiLanding";
import { ThemeProvider } from "@/contexts/ThemeContext";

export default function MoyaiPage() {
  const { accessToken } = useAuthorized();

  return (
    <ThemeProvider accessToken={accessToken}>
      <div className="flex h-screen flex-col">
        <Navbar accessToken={accessToken} isPublicPage={false} />
        <main className="min-h-0 flex-1 overflow-y-auto">
          <MoyaiLanding />
        </main>
      </div>
    </ThemeProvider>
  );
}
