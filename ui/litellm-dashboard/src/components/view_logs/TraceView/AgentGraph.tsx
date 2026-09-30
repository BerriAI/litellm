"use client";

import { useMemo, useState } from "react";

import { Button } from "@/components/ui/button";
import { cn } from "@/lib/cva.config";

import type { AgentNode, Span } from "./traceTypes";
import {
  agentsWithErrors,
  fmtMs,
  GRAPH_NODE_HEIGHT,
  invocationsOf,
  layoutAgentGraph,
  type GraphEdge,
  type GraphNode,
} from "./traceUtils";

interface AgentGraphProps {
  agents: AgentNode[];
  spans: Span[];
  /** Jump to the Tree view focused on this invocation span. */
  onOpenInvocation: (spanId: string) => void;
}

const INVOCATIONS_PAGE_SIZE = 10;

function Edge({ edge }: { edge: GraphEdge }) {
  const x1 = edge.from.x + edge.from.width;
  const y1 = edge.from.y + GRAPH_NODE_HEIGHT / 2;
  const x2 = edge.to.x;
  const y2 = edge.to.y + GRAPH_NODE_HEIGHT / 2;
  const mid = (x1 + x2) / 2;
  return (
    <g>
      <path
        d={`M${x1},${y1} C${mid},${y1} ${mid},${y2} ${x2},${y2}`}
        fill="none"
        className="stroke-violet-300 dark:stroke-violet-700"
        strokeWidth={1.5}
        markerEnd="url(#agent-graph-arrow)"
      />
      <text x={mid} y={(y1 + y2) / 2 - 6} textAnchor="middle" className="fill-muted-foreground text-[11px]">
        {edge.label}
      </text>
    </g>
  );
}

interface NodeProps {
  node: GraphNode;
  hasError: boolean;
  selected: boolean;
  onClick: (name: string) => void;
}

function Node({ node, hasError, selected, onClick }: NodeProps) {
  const { agent } = node;
  return (
    <g
      role="button"
      aria-label={`Agent ${agent.name}`}
      aria-pressed={selected}
      tabIndex={0}
      className="cursor-pointer"
      onClick={() => onClick(agent.name)}
      onKeyDown={(event) => event.key === "Enter" && onClick(agent.name)}
    >
      <rect
        x={node.x}
        y={node.y}
        width={node.width}
        height={GRAPH_NODE_HEIGHT}
        rx={8}
        strokeWidth={selected ? 2 : 1}
        className={cn(
          hasError
            ? "fill-red-50 stroke-red-400 dark:fill-red-950 dark:stroke-red-700"
            : "fill-violet-50 stroke-violet-300 dark:fill-violet-950 dark:stroke-violet-700",
          selected && "stroke-primary",
        )}
      />
      <text
        x={node.x + 10}
        y={node.y + 19}
        className={cn(
          "text-[12px] font-semibold",
          hasError ? "fill-red-700 dark:fill-red-300" : "fill-violet-700 dark:fill-violet-300",
        )}
      >
        {agent.name}
      </text>
      <text x={node.x + 10} y={node.y + 36} className="fill-muted-foreground text-[11px]">
        {`${agent.llm_calls} LLM · ${agent.tool_calls} tool`}
      </text>
    </g>
  );
}

function InvocationList({
  agentName,
  spans,
  onOpenInvocation,
}: {
  agentName: string;
  spans: Span[];
  onOpenInvocation: (spanId: string) => void;
}) {
  const [page, setPage] = useState(0);
  const invocations = useMemo(() => invocationsOf(spans, agentName), [spans, agentName]);
  const pages = Math.max(1, Math.ceil(invocations.length / INVOCATIONS_PAGE_SIZE));
  const visible = invocations.slice(page * INVOCATIONS_PAGE_SIZE, (page + 1) * INVOCATIONS_PAGE_SIZE);
  return (
    <div className="mt-3 rounded-lg border">
      <div className="flex items-center justify-between border-b px-3 py-2 text-xs font-semibold">
        <span>
          {agentName} · {invocations.length} invocation{invocations.length === 1 ? "" : "s"}
        </span>
        {pages > 1 && (
          <span className="flex items-center gap-1 font-normal">
            <Button variant="ghost" size="xs" disabled={page === 0} onClick={() => setPage(page - 1)}>
              Prev
            </Button>
            {page + 1} / {pages}
            <Button variant="ghost" size="xs" disabled={page >= pages - 1} onClick={() => setPage(page + 1)}>
              Next
            </Button>
          </span>
        )}
      </div>
      <ul aria-label={`${agentName} invocations`}>
        {visible.map((span, i) => (
          <li key={span.span_id}>
            <button
              type="button"
              onClick={() => onOpenInvocation(span.span_id)}
              className="flex w-full items-center gap-3 border-b px-3 py-1.5 text-left text-xs last:border-b-0 hover:bg-muted/60"
            >
              <span className="w-8 text-muted-foreground">#{page * INVOCATIONS_PAGE_SIZE + i + 1}</span>
              <span className="text-muted-foreground">+{fmtMs(span.start_offset_ms)}</span>
              <span>{fmtMs(span.duration_ms)}</span>
              <span className={cn("ml-auto", span.status === "error" && "text-destructive")}>
                {span.status === "error" ? "error" : "ok"}
              </span>
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}

/** One node per distinct agent, edges parent → child labelled with invocation count. */
export function AgentGraph({ agents, spans, onOpenInvocation }: AgentGraphProps) {
  const [selected, setSelected] = useState<string | null>(null);
  const layout = useMemo(() => layoutAgentGraph(agents), [agents]);
  const failing = useMemo(() => agentsWithErrors(spans), [spans]);

  if (agents.length === 0) {
    return <div className="p-4 text-sm text-muted-foreground">No agent spans in this trace.</div>;
  }

  return (
    <div className="min-h-0 flex-1 overflow-auto p-3">
      <svg
        role="img"
        aria-label="Agent graph"
        width={layout.width}
        height={layout.height}
        viewBox={`0 0 ${layout.width} ${layout.height}`}
        className="shrink-0"
      >
        <defs>
          <marker
            id="agent-graph-arrow"
            viewBox="0 0 10 10"
            refX="9"
            refY="5"
            markerWidth="6"
            markerHeight="6"
            orient="auto"
          >
            <path d="M0,0 L10,5 L0,10 z" className="fill-violet-300 dark:fill-violet-700" />
          </marker>
        </defs>
        {layout.edges.map((edge) => (
          <Edge key={`${edge.from.agent.name}->${edge.to.agent.name}`} edge={edge} />
        ))}
        {layout.nodes.map((node) => (
          <Node
            key={node.agent.name}
            node={node}
            hasError={failing.has(node.agent.name)}
            selected={selected === node.agent.name}
            onClick={(name) => setSelected(selected === name ? null : name)}
          />
        ))}
      </svg>
      <p className="mt-1 text-[11px] text-muted-foreground">Click an agent to list its invocations.</p>
      {selected && (
        <InvocationList key={selected} agentName={selected} spans={spans} onOpenInvocation={onOpenInvocation} />
      )}
    </div>
  );
}
