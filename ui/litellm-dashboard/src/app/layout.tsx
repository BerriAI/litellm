import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

import { NuqsAdapter } from "nuqs/adapters/next/app";

import AntdGlobalProvider from "@/contexts/AntdGlobalProvider";
import { AuthProvider } from "@/contexts/AuthContext";
import ReactQueryProvider from "@/contexts/ReactQueryProvider";

const inter = Inter({ subsets: ["latin"] });

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
      <body className={inter.className}>
        <NuqsAdapter>
          <ReactQueryProvider>
            <AntdGlobalProvider>
              <AuthProvider>{children}</AuthProvider>
            </AntdGlobalProvider>
          </ReactQueryProvider>
        </NuqsAdapter>
      </body>
    </html>
  );
}
