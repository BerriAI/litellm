/**
 * Individual tool item component with expandable details
 */

import { useState } from "react";
import { Typography, Tag } from "antd";
import { ToolOutlined, RightOutlined, DownOutlined } from "@ant-design/icons";
import { ParsedTool } from "./types";
import { ToolExpandedContent } from "./ToolExpandedContent";

const { Text } = Typography;

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
          <ToolOutlined style={{ color: "#8c8c8c", fontSize: 14 }} />
          <Text style={{ fontSize: 14 }}>
            {tool.index}. {tool.name}
          </Text>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <Tag color={tool.called ? "blue" : "default"}>
            {tool.called ? "called" : "not called"}
          </Tag>
          {expanded ? (
            <DownOutlined style={{ fontSize: 12, color: "#8c8c8c" }} />
          ) : (
            <RightOutlined style={{ fontSize: 12, color: "#8c8c8c" }} />
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
