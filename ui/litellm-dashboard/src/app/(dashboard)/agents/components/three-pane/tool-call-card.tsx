"use client";

/**
 * ToolCallCard — collapsible card rendering an assistant tool invocation.
 *
 * Used for `tool_call` events. The Cursor Cloud Agents UI shows these
 * collapsed by default with the tool name + a one-line preview, expanded
 * on click to show the full input.
 */
import { useState } from "react";
import { Card, Tag, Typography, Button } from "antd";

const { Text, Paragraph } = Typography;

interface ToolCallCardProps {
  tool: string;
  input: string;
  result?: string;
}

export default function ToolCallCard({ tool, input, result }: ToolCallCardProps) {
  const [expanded, setExpanded] = useState(false);
  return (
    <Card size="small" className="!mb-3" data-testid="tool-call-card" bodyStyle={{ padding: 8 }}>
      <div className="flex items-center gap-2">
        <Button size="small" type="text" onClick={() => setExpanded((v) => !v)} data-testid="tool-call-toggle">
          {expanded ? "▾" : "▸"}
        </Button>
        <Tag color="cyan" className="!m-0">
          {tool}
        </Tag>
        {!expanded && (
          <Text type="secondary" className="!truncate !text-xs">
            {input.slice(0, 80)}
          </Text>
        )}
      </div>
      {expanded && (
        <div className="mt-2 space-y-2">
          <div>
            <Text strong className="!text-xs">
              input
            </Text>
            <Paragraph className="!mb-0 whitespace-pre-wrap !text-xs !font-mono">{input}</Paragraph>
          </div>
          {result && (
            <div>
              <Text strong className="!text-xs">
                result
              </Text>
              <Paragraph className="!mb-0 whitespace-pre-wrap !text-xs !font-mono">{result}</Paragraph>
            </div>
          )}
        </div>
      )}
    </Card>
  );
}
