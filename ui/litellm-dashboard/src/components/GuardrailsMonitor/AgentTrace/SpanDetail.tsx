"use client";

import { useState } from "react";
import { Button, Tabs, Tag, Typography } from "antd";
import { CopyOutlined, DownOutlined, RightOutlined } from "@ant-design/icons";
import { Bot, MessageSquare, Wrench, Zap } from "lucide-react";
import type { Span, SpanType } from "../agentTraceTypes";

const { Text } = Typography;

const SPAN_CONFIG: Record<
  SpanType,
  { icon: React.ElementType; label: string; color: string; badgeClass: string }
> = {
  orchestrator: {
    icon: Bot,
    label: "Orchestrator",
    color: "text-blue-600",
    badgeClass: "bg-blue-100 text-blue-800 border-blue-200",
  },
  agent: {
    icon: Bot,
    label: "Agent",
    color: "text-indigo-600",
    badgeClass: "bg-indigo-100 text-indigo-800 border-indigo-200",
  },
  llm: {
    icon: MessageSquare,
    label: "LLM",
    color: "text-purple-600",
    badgeClass: "bg-purple-100 text-purple-800 border-purple-200",
  },
  mcp: {
    icon: Wrench,
    label: "MCP",
    color: "text-amber-600",
    badgeClass: "bg-amber-100 text-amber-800 border-amber-200",
  },
  function: {
    icon: Zap,
    label: "Function",
    color: "text-green-600",
    badgeClass: "bg-green-100 text-green-800 border-green-200",
  },
};

const STATUS_CONFIG: Record<
  string,
  { color: string; dotClass: string }
> = {
  success: { color: "text-green-600", dotClass: "bg-green-500" },
  error: { color: "text-red-600", dotClass: "bg-red-500" },
  running: { color: "text-blue-600", dotClass: "bg-blue-500" },
  retry: { color: "text-amber-600", dotClass: "bg-amber-500" },
};

function copyToClipboard(text: string) {
  void navigator.clipboard.writeText(text);
}

export interface SpanDetailProps {
  span: Span | null;
}

export function SpanDetail({ span }: SpanDetailProps) {
  const [metricsOpen, setMetricsOpen] = useState(true);
  const [ioOpen, setIoOpen] = useState(true);
  const [costOpen, setCostOpen] = useState(true);

  if (span === null) {
    return (
      <div className="flex flex-col items-center justify-center h-full min-h-[200px] text-slate-500 p-6 text-center">
        <p className="font-medium">Select a span</p>
        <p className="text-sm mt-1">Click a span in the trace view to see details.</p>
      </div>
    );
  }

  const config = SPAN_CONFIG[span.type];
  const statusConfig = STATUS_CONFIG[span.status] ?? STATUS_CONFIG.success;
  const Icon = config.icon;

  const tryFormatJson = (raw: string | undefined): string => {
    if (raw == null || raw === "") return "";
    try {
      return JSON.stringify(JSON.parse(raw), null, 2);
    } catch {
      return raw;
    }
  };

  return (
    <div className="flex flex-col h-full overflow-auto">
      {/* Header */}
      <div className="flex-shrink-0 border-b border-slate-200 pb-3 mb-3">
        <div className="flex items-start justify-between gap-2">
          <div className="flex items-center gap-2 min-w-0">
            <Icon className={`flex-shrink-0 w-4 h-4 ${config.color}`} />
            <span className="font-medium text-slate-900 truncate">{span.name}</span>
            <Tag className={`flex-shrink-0 ${config.badgeClass} border`}>
              {config.label}
            </Tag>
          </div>
          <Button
            type="text"
            size="small"
            icon={<CopyOutlined />}
            onClick={() => copyToClipboard(span.id)}
            title="Copy span ID"
          />
        </div>
        <div className="flex items-center gap-3 mt-2 flex-wrap">
          <span className={`flex items-center gap-1.5 text-sm ${statusConfig.color}`}>
            <span
              className={`w-2 h-2 rounded-full flex-shrink-0 ${statusConfig.dotClass}`}
            />
            {span.status}
          </span>
          <Text type="secondary" className="text-sm">
            {span.durationMs} ms
          </Text>
        </div>
      </div>

      {/* Collapsible: Metrics */}
      <div className="border border-slate-200 rounded-lg mb-3 overflow-hidden">
        <button
          type="button"
          className="w-full flex items-center justify-between px-3 py-2 bg-slate-50 hover:bg-slate-100 text-left text-sm font-medium text-slate-700 transition-colors"
          onClick={() => setMetricsOpen(!metricsOpen)}
        >
          <span>Metrics</span>
          {metricsOpen ? (
            <DownOutlined className="text-slate-500" />
          ) : (
            <RightOutlined className="text-slate-500" />
          )}
        </button>
        {metricsOpen && (
          <div className="p-3 bg-white text-sm space-y-1.5">
            <div className="flex justify-between">
              <span className="text-slate-500">Duration</span>
              <span className="font-mono">{span.durationMs} ms</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-500">Start (ms)</span>
              <span className="font-mono">{span.startMs}</span>
            </div>
            {span.tokens != null && (
              <>
                <div className="flex justify-between">
                  <span className="text-slate-500">Prompt tokens</span>
                  <span className="font-mono">{span.tokens.prompt}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">Completion tokens</span>
                  <span className="font-mono">{span.tokens.completion}</span>
                </div>
              </>
            )}
            {span.cost != null && (
              <div className="flex justify-between">
                <span className="text-slate-500">Cost</span>
                <span className="font-mono">${span.cost.toFixed(6)}</span>
              </div>
            )}
            {span.mcpServer != null && (
              <div className="flex justify-between">
                <span className="text-slate-500">MCP Server</span>
                <span className="font-mono truncate max-w-[180px]" title={span.mcpServer}>
                  {span.mcpServer}
                </span>
              </div>
            )}
            {span.mcpTool != null && (
              <div className="flex justify-between">
                <span className="text-slate-500">MCP Tool</span>
                <span className="font-mono">{span.mcpTool}</span>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Collapsible: Input / Output */}
      <div className="border border-slate-200 rounded-lg mb-3 overflow-hidden">
        <button
          type="button"
          className="w-full flex items-center justify-between px-3 py-2 bg-slate-50 hover:bg-slate-100 text-left text-sm font-medium text-slate-700 transition-colors"
          onClick={() => setIoOpen(!ioOpen)}
        >
          <span>Input / Output</span>
          {ioOpen ? (
            <DownOutlined className="text-slate-500" />
          ) : (
            <RightOutlined className="text-slate-500" />
          )}
        </button>
        {ioOpen && (
          <div className="p-2 bg-white">
            <Tabs
              size="small"
              items={[
                {
                  key: "input",
                  label: "Input",
                  children: (
                    <pre className="text-xs bg-slate-50 p-3 rounded border border-slate-200 overflow-auto max-h-64 font-mono whitespace-pre-wrap break-words">
                      {tryFormatJson(span.input) || "—"}
                    </pre>
                  ),
                },
                {
                  key: "output",
                  label: "Output",
                  children: (
                    <pre className="text-xs bg-slate-50 p-3 rounded border border-slate-200 overflow-auto max-h-64 font-mono whitespace-pre-wrap break-words">
                      {tryFormatJson(span.output) || "—"}
                    </pre>
                  ),
                },
              ]}
            />
          </div>
        )}
      </div>

      {/* Cost breakdown (LLM spans only) */}
      {span.type === "llm" && (span.cost != null || span.tokens != null) && (
        <div className="border border-slate-200 rounded-lg overflow-hidden">
          <button
            type="button"
            className="w-full flex items-center justify-between px-3 py-2 bg-slate-50 hover:bg-slate-100 text-left text-sm font-medium text-slate-700 transition-colors"
            onClick={() => setCostOpen(!costOpen)}
          >
            <span>Cost breakdown</span>
            {costOpen ? (
              <DownOutlined className="text-slate-500" />
            ) : (
              <RightOutlined className="text-slate-500" />
            )}
          </button>
          {costOpen && (
            <div className="p-3 bg-white text-sm space-y-1.5">
              {span.model != null && (
                <div className="flex justify-between">
                  <span className="text-slate-500">Model</span>
                  <span className="font-mono">{span.model}</span>
                </div>
              )}
              {span.tokens != null && (
                <div className="flex justify-between">
                  <span className="text-slate-500">Total tokens</span>
                  <span className="font-mono">
                    {span.tokens.prompt + span.tokens.completion}
                  </span>
                </div>
              )}
              {span.cost != null && (
                <div className="flex justify-between">
                  <span className="text-slate-500">Cost</span>
                  <span className="font-mono">${span.cost.toFixed(6)}</span>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
