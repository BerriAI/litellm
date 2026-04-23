import { useState } from "react";
import MessageManager from "@/components/molecules/message_manager";
import { ParsedMessage } from "./prettyMessagesTypes";
import { SectionHeader } from "./SectionHeader";
import { SimpleMessageBlock } from "./SimpleMessageBlock";

interface OutputCardProps {
  message: ParsedMessage | null;
  completionTokens?: number;
  outputCost?: number;
}

export function OutputCard({
  message,
  completionTokens,
  outputCost,
}: OutputCardProps) {
  const [isCollapsed, setIsCollapsed] = useState(false);

  const handleCopy = () => {
    if (!message) return;

    const content = message.content || "";
    navigator.clipboard.writeText(content);
    MessageManager.success("Output copied");
  };

  if (!message) {
    return (
      <div className="border border-border rounded-md overflow-hidden">
        <SectionHeader
          type="output"
          tokens={completionTokens}
          cost={outputCost}
          onCopy={handleCopy}
          isCollapsed={isCollapsed}
          onToggleCollapse={() => setIsCollapsed(!isCollapsed)}
        />
        <div
          className="overflow-hidden transition-[max-height,opacity] duration-300 ease-out"
          style={{
            maxHeight: isCollapsed ? "0px" : "10000px",
            opacity: isCollapsed ? 0 : 1,
          }}
        >
          <div className="px-4 py-3">
            <span className="text-[13px] italic text-muted-foreground">
              No response data available
            </span>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="border border-border rounded-md overflow-hidden">
      <SectionHeader
        type="output"
        tokens={completionTokens}
        cost={outputCost}
        onCopy={handleCopy}
        isCollapsed={isCollapsed}
        onToggleCollapse={() => setIsCollapsed(!isCollapsed)}
      />

      <div
        className="overflow-hidden transition-[max-height,opacity] duration-300 ease-out"
        style={{
          maxHeight: isCollapsed ? "0px" : "10000px",
          opacity: isCollapsed ? 0 : 1,
        }}
      >
        <div className="px-4 py-3">
          <SimpleMessageBlock
            label="ASSISTANT"
            content={message.content}
            toolCalls={message.toolCalls}
          />
        </div>
      </div>
    </div>
  );
}
