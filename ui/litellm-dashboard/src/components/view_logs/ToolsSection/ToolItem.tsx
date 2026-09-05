/**
 * Individual tool item component with expandable details
 */

import { useState } from "react";
import { ChevronDown, ChevronRight, Wrench } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { ParsedTool } from "./types";
import { ToolExpandedContent } from "./ToolExpandedContent";

interface ToolItemProps {
  tool: ParsedTool;
}

export function ToolItem({ tool }: ToolItemProps) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div
      style={{
        border: "1px solid #f0f0f0",
        borderRadius: 8,
        overflow: "hidden",
      }}
    >
      {/* Header Row - Always Visible */}
      <div
        onClick={() => setExpanded(!expanded)}
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          padding: "12px 16px",
          cursor: "pointer",
          background: expanded ? "#fafafa" : "#fff",
          transition: "background 0.2s",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <Wrench className="size-3.5 text-muted-foreground" />
          <span style={{ fontSize: 14 }}>
            {tool.index}. {tool.name}
          </span>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <Badge variant={tool.called ? "default" : "secondary"}>{tool.called ? "called" : "not called"}</Badge>
          {expanded ? (
            <ChevronDown className="size-3 text-muted-foreground" />
          ) : (
            <ChevronRight className="size-3 text-muted-foreground" />
          )}
        </div>
      </div>

      {/* Expanded Content */}
      {expanded && (
        <div
          style={{
            padding: "16px",
            borderTop: "1px solid #f0f0f0",
            background: "#fff",
          }}
        >
          <ToolExpandedContent tool={tool} />
        </div>
      )}
    </div>
  );
}
