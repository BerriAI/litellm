import { ToolCall } from "./prettyMessagesTypes";

interface SimpleToolCallBlockProps {
  tool: ToolCall;
  compact?: boolean;
}

export function SimpleToolCallBlock({
  tool,
  compact = false,
}: SimpleToolCallBlockProps) {
  return (
    <div
      className="bg-muted border border-border rounded-md mt-2 font-mono text-xs relative"
      style={{
        padding: compact ? "6px 10px" : "10px 14px",
      }}
    >
      <div className="absolute -top-2 left-3 bg-background px-1.5 text-[10px] text-muted-foreground border border-border rounded">
        function
      </div>

      <span className="block font-bold text-[13px] mb-1.5">{tool.name}</span>

      {Object.keys(tool.arguments).length > 0 && (
        <div>
          {Object.entries(tool.arguments).map(([key, value]) => (
            <div key={key} className="mb-0.5">
              <span className="text-xs text-muted-foreground">{key}: </span>
              <span className="text-xs">{JSON.stringify(value)}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
