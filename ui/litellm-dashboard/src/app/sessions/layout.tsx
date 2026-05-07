"use client";

import React from "react";
import AntdGlobalProvider from "@/contexts/AntdGlobalProvider";

export default function SessionsLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <AntdGlobalProvider>
      <div
        style={{
          minHeight: "100vh",
          background: "#ffffff",
          color: "#111827",
          fontFamily: "Inter, -apple-system, BlinkMacSystemFont, sans-serif",
        }}
      >
        {children}
      </div>
      <style jsx global>{`
        @import url("https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap");

        .sessions-app {
          font-family: "Inter", system-ui, sans-serif;
        }
        .sessions-app code,
        .sessions-app pre,
        .sessions-app .mono {
          font-family: "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo,
            Monaco, Consolas, monospace;
        }

        /* Markdown rendering inside chat */
        .sessions-md {
          font-size: 14px;
          line-height: 1.65;
          color: #1f2937;
        }
        .sessions-md > *:first-child {
          margin-top: 0;
        }
        .sessions-md > *:last-child {
          margin-bottom: 0;
        }
        .sessions-md h1,
        .sessions-md h2,
        .sessions-md h3,
        .sessions-md h4 {
          margin: 18px 0 8px;
          font-weight: 600;
          line-height: 1.3;
          letter-spacing: -0.01em;
        }
        .sessions-md h1 {
          font-size: 17px;
        }
        .sessions-md h2 {
          font-size: 15px;
        }
        .sessions-md h3 {
          font-size: 14px;
        }
        .sessions-md h4 {
          font-size: 13.5px;
        }
        .sessions-md p {
          margin: 0 0 12px;
        }
        .sessions-md ul,
        .sessions-md ol {
          margin: 0 0 12px;
          padding-left: 22px;
        }
        .sessions-md li {
          margin: 2px 0;
        }
        .sessions-md a {
          color: #1f2937;
          text-decoration: underline;
        }
        .sessions-md strong {
          font-weight: 600;
        }
        .sessions-md em {
          font-style: italic;
        }
        .sessions-md code {
          font-family: "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo,
            Monaco, Consolas, monospace;
          font-size: 12.5px;
          background: #f6f8fa;
          border: 1px solid #e5e7eb;
          border-radius: 4px;
          padding: 1px 5px;
        }
        .sessions-md pre {
          margin: 0 0 12px;
          padding: 12px 14px;
          background: #fcfcfc;
          border: 1px solid #e5e7eb;
          border-radius: 8px;
          overflow: auto;
          font-family: "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo,
            Monaco, Consolas, monospace;
          font-size: 12.5px;
          line-height: 1.55;
        }
        .sessions-md pre code {
          background: transparent;
          border: none;
          padding: 0;
          font-size: inherit;
        }
        .sessions-md blockquote {
          margin: 0 0 12px;
          padding: 4px 12px;
          border-left: 2px solid #d1d5db;
          color: #6b7280;
        }
      `}</style>
    </AntdGlobalProvider>
  );
}
