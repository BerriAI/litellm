"use client";

import { useState, useMemo } from "react";
import { Drawer } from "antd";
import type { AgentTraceSession, Span } from "../agentTraceTypes";
import { TraceView } from "./TraceView";
import { SpanDetail } from "./SpanDetail";

function findSpan(spans: Span[], id: string): Span | null {
  for (const s of spans) {
    if (s.id === id) return s;
    if (s.children?.length) {
      const found = findSpan(s.children, id);
      if (found) return found;
    }
  }
  return null;
}

export interface AgentTraceDrawerProps {
  open: boolean;
  onClose: () => void;
  session: AgentTraceSession | null;
}

const DRAWER_WIDTH = 960;

export function AgentTraceDrawer({
  open,
  onClose,
  session,
}: AgentTraceDrawerProps) {
  const [selectedSpanId, setSelectedSpanId] = useState<string | null>(null);

  const selectedSpan = useMemo(() => {
    if (!session?.spans || selectedSpanId == null) return null;
    return findSpan(session.spans, selectedSpanId);
  }, [session?.spans, selectedSpanId]);

  const handleClose = () => {
    setSelectedSpanId(null);
    onClose();
  };

  return (
    <Drawer
      title={session ? session.rootAgentName : "Agent Trace"}
      placement="right"
      width={Math.min(DRAWER_WIDTH, typeof window !== "undefined" ? window.innerWidth * 0.9 : 960)}
      open={open}
      onClose={handleClose}
      destroyOnClose
      styles={{ body: { padding: 0, display: "flex", flexDirection: "column", height: "100%" } }}
    >
      {session == null ? (
        <div className="flex flex-col items-center justify-center flex-1 text-slate-500 p-6 text-center">
          <p className="text-sm">No session selected</p>
        </div>
      ) : (
        <div className="flex flex-1 min-h-0">
          <div className="flex-1 min-w-0 border-r border-slate-200 overflow-hidden flex flex-col p-4">
            <TraceView
              session={session}
              selectedSpanId={selectedSpanId}
              onSelectSpan={setSelectedSpanId}
            />
          </div>
          <div className="w-[380px] flex-shrink-0 overflow-hidden flex flex-col p-4 bg-slate-50/50">
            <SpanDetail span={selectedSpan} />
          </div>
        </div>
      )}
    </Drawer>
  );
}
