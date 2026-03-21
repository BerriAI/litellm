/**
 * Tools section component that displays all available tools from the request
 * and indicates which ones were actually called in the response
 */

import { Collapse, Typography } from "antd";
import { LogEntry } from "../columns";
import { parseToolsFromLog } from "./utils";
import { ToolItem } from "./ToolItem";

const { Text } = Typography;

interface ToolsSectionProps {
  log: LogEntry;
}

export function ToolsSection({ log }: ToolsSectionProps) {
  const tools = parseToolsFromLog(log);

  // Don't render if no tools
  if (tools.length === 0) return null;

  // Calculate summary stats
  const totalTools = tools.length;
  const calledTools = tools.filter((t) => t.called).length;
  
  // Get preview of first 2 tool names
  const toolNamePreview = tools
    .slice(0, 2)
    .map((t) => t.name)
    .join(", ");
  const hasMoreTools = tools.length > 2;

  return (
    <div className="bg-white rounded-lg shadow w-full max-w-full overflow-hidden mb-6">
      <Collapse
        expandIconPosition="start"
        items={[
          {
            key: "1",
            label: (
              <div style={{ display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
                <h3 className="text-lg font-medium text-gray-900">Tools</h3>
                <Text type="secondary" style={{ fontSize: 14 }}>
                  {totalTools} provided, {calledTools} called
                </Text>
                <Text type="secondary" style={{ fontSize: 14 }}>
                  • {toolNamePreview}
                  {hasMoreTools && "..."}
                </Text>
              </div>
            ),
            children: (
              <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                {tools.map((tool) => (
                  <ToolItem key={tool.name} tool={tool} />
                ))}
              </div>
            ),
          },
        ]}
      />
    </div>
  );
}
