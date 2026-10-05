/**
 * JSON view of tool definition
 */

import { ParsedTool } from "./types";

interface JsonToolViewProps {
  tool: ParsedTool;
}

export function JsonToolView({ tool }: JsonToolViewProps) {
  // Reconstruct the original tool definition
  const toolJson = {
    type: "function",
    function: {
      name: tool.name,
      description: tool.description,
      parameters: tool.parameters,
    },
  };

  return (
    <pre className="m-0 max-h-[300px] overflow-auto whitespace-pre-wrap break-words rounded bg-muted p-3 text-xs text-foreground">
      {JSON.stringify(toolJson, null, 2)}
    </pre>
  );
}
