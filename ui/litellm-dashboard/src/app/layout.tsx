import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

import { NuqsAdapter } from "nuqs/adapters/next/app";

import { AuthProvider } from "@/contexts/AuthContext";
import ReactQueryProvider from "@/contexts/ReactQueryProvider";
import { Toaster } from "@/components/ui/sonner";

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
            <AuthProvider>{children}</AuthProvider>
            <Toaster />
          </ReactQueryProvider>
        </NuqsAdapter>
      </body>
    </html>
  );
}
