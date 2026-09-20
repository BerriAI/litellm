import { useState } from "react";
import { toast } from "@/lib/toast";
import { ResponsesWebSocketTurn } from "./prettyMessagesTypes";
import { SectionHeader } from "./SectionHeader";
import { SimpleMessageBlock } from "./SimpleMessageBlock";

interface ResponsesWebSocketPrettyViewProps {
  turns: readonly ResponsesWebSocketTurn[];
  completionTokens?: number;
  outputCost?: number;
}

export function ResponsesWebSocketPrettyView({
  turns,
  completionTokens,
  outputCost,
}: ResponsesWebSocketPrettyViewProps) {
  const [isCollapsed, setIsCollapsed] = useState(false);
  const copyOutput = async () => {
    await navigator.clipboard.writeText(JSON.stringify(turns, null, 2));
    toast.success("Output copied");
  };
  return (
    <div className="overflow-hidden rounded-md border border-border">
      <SectionHeader
        type="output"
        tokens={completionTokens}
        cost={outputCost}
        onCopy={copyOutput}
        isCollapsed={isCollapsed}
        onToggleCollapse={() => setIsCollapsed(!isCollapsed)}
      />
      {!isCollapsed && (
        <div className="divide-y divide-border">
          {turns.map((turn, index) => (
            <section
              key={`${turn.id}-${index}`}
              className="px-4 py-3"
              aria-label={`Turn ${index + 1} · ${turn.status}`}
            >
              <h4 className="mb-2 text-sm font-medium">
                Turn {index + 1} · {turn.status}
              </h4>
              {turn.id && <div className="mb-2 break-all text-xs text-muted-foreground">{turn.id}</div>}
              {turn.detail && <p className="mb-2 whitespace-pre-wrap text-sm">{turn.detail}</p>}
              {turn.message ? (
                <SimpleMessageBlock
                  label="ASSISTANT"
                  content={turn.message.content}
                  toolCalls={turn.message.toolCalls}
                />
              ) : (
                <span className="text-[13px] italic text-muted-foreground">No output recorded for this turn</span>
              )}
            </section>
          ))}
        </div>
      )}
    </div>
  );
}
