"use client";

import React from "react";
import AntdGlobalProvider from "@/contexts/AntdGlobalProvider";

/**
 * Standalone layout for v2 sessions UI.
 *
 * Sessions-led parent for the managed agents experience. Same minimal
 * monochrome dev-tool surface as /agents — no admin shell.
 */
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
          fontFamily:
            '-apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif',
          fontSize: 13,
          letterSpacing: "-0.005em",
        }}
      >
        {children}
      </div>
      <style jsx global>{`
        :root {
          --bg-page: #ffffff;
          --bg-rail: #fafafa;
          --bg-code: #f6f8fa;
          --bg-hover: #f1f3f5;
          --bg-selected: #ebedf0;
          --border-color: #ececf0;
          --border-strong: #d1d5db;
          --text-primary: #1f1f1f;
          --text-secondary: #5e5e5e;
          --text-muted: #9b9ba0;
          --mono-font: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas,
            monospace;
        }

        .sessions-mono {
          font-family: var(--mono-font);
          font-feature-settings: "liga" 0;
          font-variant-numeric: tabular-nums;
        }

        .sessions-header {
          height: 40px;
          border-bottom: 1px solid var(--border-color);
          display: flex;
          align-items: center;
          padding: 0 14px;
          background: var(--bg-page);
          gap: 10px;
        }

        .sessions-header-title {
          font-size: 13px;
          font-weight: 500;
          color: var(--text-primary);
        }

        .sessions-header-right {
          margin-left: auto;
          display: flex;
          align-items: center;
          gap: 8px;
        }

        .sessions-btn {
          font-size: 12px;
          line-height: 1;
          padding: 6px 10px;
          border: 1px solid var(--border-color);
          background: var(--bg-page);
          color: var(--text-primary);
          cursor: pointer;
          border-radius: 4px;
          font-family: inherit;
        }
        .sessions-btn:hover {
          background: var(--bg-hover);
          border-color: var(--border-strong);
        }
        .sessions-btn:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }

        .sessions-chip {
          display: inline-flex;
          align-items: center;
          height: 26px;
          padding: 0 10px;
          font-size: 12px;
          color: var(--text-secondary);
          background: var(--bg-page);
          border: 1px solid var(--border-color);
          border-radius: 13px;
          cursor: pointer;
          white-space: nowrap;
          font-family: inherit;
        }
        .sessions-chip:hover {
          background: var(--bg-hover);
          border-color: var(--border-strong);
        }
        .sessions-chip[data-active="true"] {
          background: #111827;
          color: #ffffff;
          border-color: #111827;
        }

        .sessions-table {
          width: 100%;
          border-collapse: collapse;
          font-size: 12px;
        }
        .sessions-table thead th {
          text-align: left;
          font-weight: 500;
          font-size: 10px;
          letter-spacing: 0.6px;
          text-transform: uppercase;
          color: var(--text-secondary);
          padding: 8px 16px;
          border-bottom: 1px solid var(--border-color);
        }
        .sessions-table tbody td {
          padding: 10px 16px;
          border-bottom: 1px solid var(--border-color);
          vertical-align: middle;
        }
        .sessions-table tbody tr {
          cursor: pointer;
        }
        .sessions-table tbody tr:hover {
          background: var(--bg-hover);
        }

        .sessions-status-dot {
          display: inline-block;
          width: 6px;
          height: 6px;
          border-radius: 50%;
          margin-right: 8px;
          vertical-align: middle;
          flex-shrink: 0;
        }

        .sessions-link-back {
          color: var(--text-secondary);
          text-decoration: none;
          font-size: 12px;
        }
        .sessions-link-back:hover {
          color: var(--text-primary);
        }

        .sessions-rail-section {
          padding: 14px 12px 4px;
          font-size: 10.5px;
          color: var(--text-muted);
          letter-spacing: 0.02em;
        }

        .sessions-rail-row {
          display: block;
          width: 100%;
          text-align: left;
          padding: 5px 12px;
          border: none;
          border-radius: 6px;
          background: transparent;
          cursor: pointer;
          outline: none;
          color: var(--text-primary);
          font-family: inherit;
          font-size: 12.5px;
          line-height: 1.45;
          margin: 0 6px;
          width: calc(100% - 12px);
        }
        .sessions-rail-row:hover {
          background: var(--bg-hover);
        }
        .sessions-rail-row[data-active="true"] {
          background: var(--bg-selected);
          font-weight: 500;
        }

        /* Markdown rendering — tighter, more refined */
        .sessions-md {
          font-size: 13.5px;
          line-height: 1.6;
          color: var(--text-primary);
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
          margin: 20px 0 6px;
          font-weight: 600;
          line-height: 1.3;
          color: var(--text-primary);
          letter-spacing: -0.01em;
        }
        .sessions-md h1 {
          font-size: 17px;
        }
        .sessions-md h2 {
          font-size: 15px;
        }
        .sessions-md h3 {
          font-size: 13.5px;
        }
        .sessions-md h4 {
          font-size: 13px;
        }
        .sessions-md p {
          margin: 0 0 10px;
        }
        .sessions-md ul,
        .sessions-md ol {
          margin: 0 0 10px;
          padding-left: 20px;
        }
        .sessions-md li {
          margin: 1px 0;
        }
        .sessions-md a {
          color: var(--text-primary);
          text-decoration: underline;
        }
        .sessions-md strong {
          font-weight: 600;
        }
        .sessions-md em {
          font-style: italic;
        }
        .sessions-md code {
          font-family: var(--mono-font);
          font-size: 12.5px;
          background: var(--bg-code);
          border: 1px solid var(--border-color);
          border-radius: 3px;
          padding: 1px 5px;
        }
        .sessions-md pre {
          margin: 0 0 12px;
          padding: 12px 14px;
          background: var(--bg-code);
          border: 1px solid var(--border-color);
          border-radius: 6px;
          overflow: auto;
          font-family: var(--mono-font);
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
          border-left: 2px solid var(--border-strong);
          color: var(--text-secondary);
        }
        .sessions-md table {
          border-collapse: collapse;
          margin: 0 0 12px;
          font-size: 12.5px;
        }
        .sessions-md th,
        .sessions-md td {
          border: 1px solid var(--border-color);
          padding: 6px 10px;
          text-align: left;
        }
        .sessions-md th {
          background: var(--bg-rail);
          font-weight: 600;
        }

        .sessions-composer-wrap {
          border: 1px solid var(--border-color);
          border-radius: 10px;
          background: var(--bg-page);
          padding: 12px 14px 10px;
          transition: border-color 120ms ease;
        }
        .sessions-composer-wrap:focus-within {
          border-color: var(--border-strong);
        }

        .sessions-composer-textarea {
          width: 100%;
          border: none;
          outline: none;
          resize: none;
          font-family: inherit;
          font-size: 13.5px;
          line-height: 1.5;
          color: var(--text-primary);
          background: transparent;
          padding: 0;
        }
        .sessions-composer-textarea::placeholder {
          color: var(--text-muted);
        }

        .sessions-stop-btn {
          width: 22px;
          height: 22px;
          display: inline-flex;
          align-items: center;
          justify-content: center;
          border: none;
          background: var(--text-primary);
          color: #ffffff;
          border-radius: 5px;
          cursor: pointer;
          padding: 0;
        }
        .sessions-stop-btn:hover {
          background: #000000;
        }
        .sessions-stop-btn:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }
        .sessions-stop-btn svg {
          display: block;
        }
      `}</style>
    </AntdGlobalProvider>
  );
}
