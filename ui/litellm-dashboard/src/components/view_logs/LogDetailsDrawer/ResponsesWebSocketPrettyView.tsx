import { useState } from "react";
import { toast } from "@/lib/toast";
import { ResponsesWebSocketTurn } from "./prettyMessagesTypes";
import { SectionHeader } from "./SectionHeader";
import { SimpleMessageBlock } from "./SimpleMessageBlock";
import { Button } from "@/components/ui/button";

const TURNS_PER_PAGE = 50;

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
  const [page, setPage] = useState(0);
  const lastPage = Math.max(0, Math.ceil(turns.length / TURNS_PER_PAGE) - 1);
  const currentPage = Math.min(page, lastPage);
  const start = currentPage * TURNS_PER_PAGE;
  const end = Math.min(start + TURNS_PER_PAGE, turns.length);
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
          {turns.slice(start, end).map((turn, index) => (
            <section
              key={`${turn.id}-${start + index}`}
              className="px-4 py-3"
              aria-label={`Turn ${start + index + 1} · ${turn.status}`}
            >
              <h4 className="mb-2 text-sm font-medium">
                Turn {start + index + 1} · {turn.status}
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
          {turns.length > TURNS_PER_PAGE && (
            <nav aria-label="Response turns" className="flex items-center justify-between gap-2 px-4 py-3">
              <Button variant="outline" size="sm" disabled={currentPage === 0} onClick={() => setPage(currentPage - 1)}>
                Previous turns
              </Button>
              <span className="text-sm" aria-live="polite">
                Turns {start + 1}–{end} of {turns.length}
              </span>
              <Button
                variant="outline"
                size="sm"
                disabled={currentPage === lastPage}
                onClick={() => setPage(currentPage + 1)}
              >
                Next turns
              </Button>
            </nav>
          )}
        </div>
      )}
    </div>
  );
}
