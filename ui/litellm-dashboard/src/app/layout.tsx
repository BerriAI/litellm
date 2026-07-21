import type { Metadata } from "next";
import "./globals.css";

import AntdGlobalProvider from "@/contexts/AntdGlobalProvider";
import { AuthProvider } from "@/contexts/AuthContext";
import ReactQueryProvider from "@/contexts/ReactQueryProvider";

export const metadata: Metadata = {
  title: "LiteLLM Dashboard",
  description: "LiteLLM Proxy Admin UI",
  icons: { icon: "/get_favicon" },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body style={{ fontFamily: "system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif" }}>
        <ReactQueryProvider>
          <AntdGlobalProvider>
            <AuthProvider>{children}</AuthProvider>
          </AntdGlobalProvider>
        </ReactQueryProvider>
      </body>
    </html>
  );
}
