"use client";

import { useState, useMemo } from "react";
import { ChevronDown, ChevronRight, Clock, Cpu } from "lucide-react";
import { Bot, MessageSquare, Wrench, Zap } from "lucide-react";
import type { AgentTraceSession, Span, SpanType } from "../agentTraceTypes";

export interface TraceViewProps {
  session: AgentTraceSession;
  selectedSpanId: string | null;
  onSelectSpan: (id: string) => void;
}

const SPAN_CONFIG: Record<
  SpanType,
  { icon: React.ElementType; label: string; bg: string; text: string; border: string; bar: string }
> = {
  orchestrator: {
    icon: Bot,
    label: "Orchestrator",
    bg: "bg-blue-50",
    text: "text-blue-700",
    border: "border-blue-200",
    bar: "bg-blue-500",
  },
  agent: {
    icon: Bot,
    label: "Agent",
    bg: "bg-indigo-50",
    text: "text-indigo-700",
    border: "border-indigo-200",
    bar: "bg-indigo-500",
  },
  llm: {
    icon: MessageSquare,
    label: "LLM",
    bg: "bg-purple-50",
    text: "text-purple-700",
    border: "border-purple-200",
    bar: "bg-purple-500",
  },
  mcp: {
    icon: Wrench,
    label: "MCP",
    bg: "bg-amber-50",
    text: "text-amber-700",
    border: "border-amber-200",
    bar: "bg-amber-500",
  },
  function: {
    icon: Zap,
    label: "Function",
    bg: "bg-green-50",
    text: "text-green-700",
    border: "border-green-200",
    bar: "bg-green-500",
  },
};

function formatDuration(ms: number): string {
  if (ms >= 1000) return `${(ms / 1000).toFixed(3)}s`;
  return `${ms}ms`;
}

function flattenSpans(
  spans: Span[],
  depth = 0
): Array<{ span: Span; depth: number }> {
  const result: Array<{ span: Span; depth: number }> = [];
  for (const span of spans) {
    result.push({ span, depth });
    if (span.children?.length) {
      result.push(...flattenSpans(span.children, depth + 1));
    }
  }
  return result;
}

function TreeRow({
  span,
  depth,
  isSelected,
  isExpanded,
  hasChildren,
  onSelect,
  onToggle,
}: {
  span: Span;
  depth: number;
  isSelected: boolean;
  isExpanded: boolean;
  hasChildren: boolean;
  onSelect: () => void;
  onToggle: () => void;
}) {
  const config = SPAN_CONFIG[span.type];
  const Icon = config.icon;
  return (
    <div
      role="button"
      tabIndex={0}
      onClick={onSelect}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          onSelect();
        }
      }}
      className={`flex items-center gap-1.5 px-2 py-1.5 cursor-pointer border-b border-slate-100 transition-colors ${isSelected ? "bg-blue-50" : "hover:bg-slate-50"}`}
      style={{ paddingLeft: `${8 + depth * 20}px` }}
    >
      <button
        type="button"
        onClick={(e) => {
          e.stopPropagation();
          onToggle();
        }}
        className="w-4 h-4 flex items-center justify-center text-slate-400 hover:text-slate-600 flex-shrink-0"
        aria-label={isExpanded ? "Collapse" : "Expand"}
      >
        {hasChildren ? (
          isExpanded ? (
            <ChevronDown className="w-3 h-3" />
          ) : (
            <ChevronRight className="w-3 h-3" />
          )
        ) : (
          <span className="w-3 h-3 block" />
        )}
      </button>
      <div
        className={`w-5 h-5 rounded flex items-center justify-center flex-shrink-0 ${config.bg} border ${config.border}`}
      >
        <Icon className={`w-2.5 h-2.5 ${config.text}`} />
      </div>
      <span
        className={`text-xs font-mono flex-shrink-0 truncate max-w-[140px] ${isSelected ? "text-slate-900" : "text-slate-700"}`}
      >
        {span.name}
      </span>
      <span
        className={`text-[10px] px-1 py-0.5 rounded border flex-shrink-0 ${config.bg} ${config.text} ${config.border}`}
      >
        {config.label}
      </span>
      <div className="flex-1 min-w-0" />
      {span.tokens != null && (
        <div className="flex items-center gap-1 text-[10px] text-slate-400 font-mono flex-shrink-0">
          <Cpu className="w-2.5 h-2.5" />
          <span>{span.tokens.prompt + span.tokens.completion} tok</span>
        </div>
      )}
      {span.cost != null && (
        <span className="text-[10px] text-slate-400 font-mono flex-shrink-0">
          ${span.cost.toFixed(5)}
        </span>
      )}
      <span className="text-[10px] text-slate-500 font-mono flex-shrink-0 w-14 text-right">
        {formatDuration(span.durationMs)}
      </span>
      <span
        className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${span.status === "success" ? "bg-green-500" : span.status === "error" ? "bg-red-500" : "bg-amber-500"}`}
        aria-hidden
      />
    </div>
  );
}

function TreeView({
  session,
  selectedSpanId,
  onSelectSpan,
}: TraceViewProps) {
  const [expandedIds, setExpandedIds] = useState<Set<string>>(() => {
    const ids = new Set<string>();
    function collect(spans: Span[]) {
      for (const s of spans) {
        ids.add(s.id);
        if (s.children?.length) collect(s.children);
      }
    }
    collect(session.spans);
    return ids;
  });

  const toggleExpand = (id: string) => {
    setExpandedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const renderSpans = (spans: Span[], depth: number): React.ReactNode[] => {
    const rows: React.ReactNode[] = [];
    for (const span of spans) {
      const isExpanded = expandedIds.has(span.id);
      const hasChildren = !!(span.children && span.children.length > 0);
      rows.push(
        <TreeRow
          key={span.id}
          span={span}
          depth={depth}
          isSelected={selectedSpanId === span.id}
          isExpanded={isExpanded}
          hasChildren={hasChildren}
          onSelect={() => onSelectSpan(span.id)}
          onToggle={() => toggleExpand(span.id)}
        />
      );
      if (hasChildren && isExpanded && span.children) {
        rows.push(...renderSpans(span.children, depth + 1));
      }
    }
    return rows;
  };

  if (!session.spans.length) {
    return (
      <div className="flex-1 flex items-center justify-center text-slate-500 text-sm">
        No trace data available
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-y-auto flex flex-col min-h-0">
      <div className="flex items-center px-2 py-1.5 border-b border-slate-200 bg-slate-50 sticky top-0 z-10 shrink-0">
        <div className="w-4 shrink-0" />
        <div className="w-5 shrink-0 ml-1.5" />
        <span className="text-[10px] text-slate-500 uppercase tracking-wider ml-1.5">
          Span
        </span>
        <div className="flex-1" />
        <span className="text-[10px] text-slate-500 uppercase tracking-wider w-20 text-right">
          Tokens
        </span>
        <span className="text-[10px] text-slate-500 uppercase tracking-wider w-16 text-right">
          Cost
        </span>
        <span className="text-[10px] text-slate-500 uppercase tracking-wider w-14 text-right">
          Duration
        </span>
        <div className="w-4 shrink-0 ml-1.5" />
      </div>
      <div className="shrink-0">{renderSpans(session.spans, 0)}</div>
    </div>
  );
}

function WaterfallView({
  session,
  selectedSpanId,
  onSelectSpan,
}: TraceViewProps) {
  const totalMs = session.totalDurationMs;
  const flatSpans = useMemo(() => flattenSpans(session.spans), [session.spans]);

  if (!flatSpans.length) {
    return (
      <div className="flex-1 flex items-center justify-center text-slate-500 text-sm">
        No trace data available
      </div>
    );
  }

  const LABEL_WIDTH = 200;
  const tickCount = 6;
  const ticks = Array.from(
    { length: tickCount + 1 },
    (_, i) => Math.round((totalMs / tickCount) * i)
  );

  return (
    <div className="flex-1 overflow-auto flex flex-col min-h-0">
      <div className="flex border-b border-slate-200 bg-slate-50 sticky top-0 z-10 shrink-0">
        <div
          style={{ width: LABEL_WIDTH }}
          className="shrink-0 px-3 py-1.5"
        >
          <span className="text-[10px] text-slate-500 uppercase tracking-wider">
            Span
          </span>
        </div>
        <div className="flex-1 relative py-1.5 pr-4">
          <div className="relative h-4">
            {ticks.map((tick) => (
              <div
                key={tick}
                className="absolute top-0 flex flex-col items-center"
                style={{ left: `${(tick / totalMs) * 100}%` }}
              >
                <span className="text-[9px] text-slate-400 font-mono -translate-x-1/2">
                  {tick >= 1000 ? `${(tick / 1000).toFixed(1)}s` : `${tick}ms`}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>
      {flatSpans.map(({ span, depth }) => {
        const config = SPAN_CONFIG[span.type];
        const Icon = config.icon;
        const leftPct = (span.startMs / totalMs) * 100;
        const widthPct = Math.max((span.durationMs / totalMs) * 100, 0.5);
        const isSelected = selectedSpanId === span.id;
        return (
          <div
            key={span.id}
            role="button"
            tabIndex={0}
            onClick={() => onSelectSpan(span.id)}
            onKeyDown={(e) => {
              if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                onSelectSpan(span.id);
              }
            }}
            className={`flex items-center border-b border-slate-100 cursor-pointer transition-colors ${isSelected ? "bg-blue-50" : "hover:bg-slate-50"}`}
          >
            <div
              style={{
                width: LABEL_WIDTH,
                paddingLeft: `${12 + depth * 16}px`,
              }}
              className="shrink-0 flex items-center gap-1.5 py-1.5 pr-3"
            >
              <div
                className={`w-4 h-4 rounded flex items-center justify-center shrink-0 ${config.bg} border ${config.border}`}
              >
                <Icon className={`w-2 h-2 ${config.text}`} />
              </div>
              <span
                className={`text-[11px] font-mono truncate ${isSelected ? "text-slate-900" : "text-slate-600"}`}
              >
                {span.name}
              </span>
            </div>
            <div className="flex-1 relative py-2 pr-4 min-h-[24px]">
              {ticks.slice(1, -1).map((tick) => (
                <div
                  key={tick}
                  className="absolute top-0 bottom-0 w-px bg-slate-200"
                  style={{ left: `${(tick / totalMs) * 100}%` }}
                />
              ))}
              <div
                className={`absolute top-1.5 bottom-1.5 rounded-sm ${config.bar} ${isSelected ? "opacity-90" : "opacity-70"} transition-opacity`}
                style={{
                  left: `${leftPct}%`,
                  width: `${widthPct}%`,
                  minWidth: "3px",
                }}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
}

export function TraceView({
  session,
  selectedSpanId,
  onSelectSpan,
}: TraceViewProps) {
  const [activeTab, setActiveTab] = useState<"tree" | "waterfall">("tree");

  const totalTokens = useMemo(() => {
    let t = 0;
    function sum(spans: Span[]) {
      for (const s of spans) {
        if (s.tokens) t += s.tokens.prompt + s.tokens.completion;
        if (s.children?.length) sum(s.children);
      }
    }
    sum(session.spans);
    return t;
  }, [session.spans]);

  return (
    <div className="flex flex-col h-full min-h-0">
      <div className="shrink-0 px-2 py-2 border-b border-slate-200 space-y-2">
        <div className="flex items-center justify-between gap-2">
          <h3 className="text-sm font-semibold text-slate-900 truncate">
            {session.rootAgentName}
          </h3>
          <span
            className={`text-[10px] px-2 py-0.5 rounded-full font-medium shrink-0 ${session.status === "success" ? "bg-green-50 text-green-700 border border-green-200" : "bg-red-50 text-red-700 border border-red-200"}`}
          >
            {session.status}
          </span>
        </div>
        <div className="font-mono text-[11px] text-slate-500 truncate">
          {session.shortId}
        </div>
        <div className="flex items-center gap-3 flex-wrap text-xs text-slate-600">
          <span className="flex items-center gap-1">
            <Clock className="w-3 h-3 text-slate-400" />
            {formatDuration(session.totalDurationMs)}
          </span>
          <span className="flex items-center gap-1">
            <Cpu className="w-3 h-3 text-slate-400" />
            {totalTokens > 0
              ? `${totalTokens} tokens`
              : `${session.totalSpans} spans`}
          </span>
          <span className="font-mono">${session.totalCost.toFixed(4)}</span>
          <span className="text-slate-400">{session.relativeTime}</span>
        </div>
      </div>

      <div className="flex gap-0 border-b border-slate-200 shrink-0">
        {(["tree", "waterfall"] as const).map((tab) => (
          <button
            key={tab}
            type="button"
            onClick={() => setActiveTab(tab)}
            className={`relative px-3 py-2 text-xs font-medium capitalize transition-colors ${activeTab === tab ? "text-blue-600" : "text-slate-500 hover:text-slate-700"}`}
          >
            {tab === "tree" ? "Tree View" : "Waterfall"}
            {activeTab === tab && (
              <span
                className="absolute bottom-0 left-0 right-0 h-0.5 bg-blue-500"
                aria-hidden
              />
            )}
          </button>
        ))}
      </div>

      <div className="flex-1 min-h-0 overflow-hidden">
        {activeTab === "tree" ? (
          <TreeView
            session={session}
            selectedSpanId={selectedSpanId}
            onSelectSpan={onSelectSpan}
          />
        ) : (
          <WaterfallView
            session={session}
            selectedSpanId={selectedSpanId}
            onSelectSpan={onSelectSpan}
          />
        )}
      </div>
    </div>
  );
}
