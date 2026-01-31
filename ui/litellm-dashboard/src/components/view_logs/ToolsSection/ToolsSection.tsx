/**
 * Tools section component that displays all available tools from the request
 * and indicates which ones were actually called in the response
 */

import { Typography } from "antd";
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

  return (
    <div className="bg-white rounded-lg shadow w-full max-w-full overflow-hidden p-4 mb-6">
      <Text
        strong
        style={{
          display: "block",
          marginBottom: 12,
          fontSize: 16,
        }}
      >
        Tools
      </Text>
      <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
        {tools.map((tool) => (
          <ToolItem key={tool.name} tool={tool} />
        ))}
      </div>
    </div>
  );
}
