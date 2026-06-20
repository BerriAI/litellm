import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

import AntdGlobalProvider from "@/contexts/AntdGlobalProvider";
import ReactQueryProvider from "@/contexts/ReactQueryProvider";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "LiteLLM Dashboard",
  description: "LiteLLM Proxy Admin UI",
  icons: { icon: "./favicon.ico" },
};

// Runs before hydration to set the dark class from the stored preference, avoiding a flash of light mode
const themeScript = `
  (function() {
    try {
      var stored = localStorage.getItem('litellm-dark-mode');
      var isDark = stored === null
        ? window.matchMedia('(prefers-color-scheme: dark)').matches
        : stored === 'true';
      if (isDark) {
        document.documentElement.classList.add('dark');
      }
    } catch (e) {}
  })();
`;

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        {/* Static, developer-controlled theme bootstrap; must run before paint to prevent a flash of light mode */}
        {/* eslint-disable-next-line react/no-danger */}
        <script dangerouslySetInnerHTML={{ __html: themeScript }} />
      </head>
      <body className={inter.className}>
        <ReactQueryProvider>
          <AntdGlobalProvider>{children}</AntdGlobalProvider>
        </ReactQueryProvider>
      </body>
    </html>
  );
}
