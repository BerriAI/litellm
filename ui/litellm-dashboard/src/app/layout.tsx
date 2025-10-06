import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { FeatureFlagsProvider } from "@/hooks/useFeatureFlags";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "LiteLLM Dashboard",
  description: "LiteLLM Proxy Admin UI",
  icons: { icon: "./favicon.ico" },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <FeatureFlagsProvider>
        <body className={inter.className}>{children}</body>
      </FeatureFlagsProvider>
    </html>
  );
}
