import { useEffect, type RefObject } from "react";
import { ArrowDown, Check, ChevronDown } from "lucide-react";
import { MessageScroller, useMessageScroller } from "@shadcn/react/message-scroller";
import { ChatMessageContent } from "@/components/chat/ChatMessages";
import CopyButton from "@/components/shared/CopyButton";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { Skeleton } from "@/components/ui/skeleton";
import { Table, TableBody, TableCell, TableRow } from "@/components/ui/table";
import type { ActionEntry, ConversationEntry } from "./useLiteAdmin";
import { AriaAvatar } from "./AriaAvatar";
import { ARIA_AGENT_LABEL } from "./aria-brand";

const ACTION_STATUS: Record<ActionEntry["status"], string> = {
  review: "Review change",
  applying: "Applying…",
  completed: "Completed",
  cancelled: "Cancelled",
  unknown: "Check result",
};

const SUGGESTED_QUESTIONS = [
  "How much have I spent this week?",
  "Which models are costing the most?",
  "How can I reduce my spend?",
];

interface ConversationProps {
  entries: ConversationEntry[];
  thinking: boolean;
  readOnly?: boolean;
  open: boolean;
  reviewRef: RefObject<HTMLDivElement | null>;
  onAnswer: (id: string, approved: boolean) => void;
  onQuestion: (question: string) => void;
  questionsDisabled: boolean;
}

export function LiteAdminConversation({
  entries,
  thinking,
  open,
  readOnly = false,
  reviewRef,
  onAnswer,
  onQuestion,
  questionsDisabled,
}: ConversationProps) {
  return (
    <MessageScroller.Provider autoScroll>
      <MessageScroller.Root className="relative flex min-h-0 flex-1 flex-col overflow-hidden">
        <MessageScroller.Viewport
          className="min-h-0 flex-1 overflow-y-auto overscroll-contain data-pending-scroll:invisible"
          aria-label={`${ARIA_AGENT_LABEL} conversation`}
        >
          <MessageScroller.Content className="flex flex-col gap-4 p-4" aria-busy={thinking}>
            {entries.map((entry) => (
              <MessageScroller.Item key={entry.id} messageId={entry.id}>
                {entry.kind === "message" && (
                  <ChatMessageContent
                    message={entry.message}
                    allowImages={false}
                    isLastMessage={false}
                    isStreaming={false}
                  />
                )}
                {entry.kind === "action" && (
                  <ActionCard entry={entry} open={open} reviewRef={reviewRef} onAnswer={onAnswer} />
                )}
                {entry.kind === "error" && (
                  <Alert variant="destructive">
                    <AlertDescription>{entry.text}</AlertDescription>
                  </Alert>
                )}
              </MessageScroller.Item>
            ))}
            {thinking && (
              <MessageScroller.Item messageId="thinking">
                <div role="status" aria-label={`${ARIA_AGENT_LABEL} is working`} className="space-y-2">
                  <Skeleton className="h-3 w-3/4" />
                  <Skeleton className="h-3 w-1/2" />
                  <span className="sr-only">Working…</span>
                </div>
              </MessageScroller.Item>
            )}
          </MessageScroller.Content>
        </MessageScroller.Viewport>
        {entries.length === 0 && (
          <div className="absolute inset-0 flex flex-col items-center gap-4 overflow-y-auto p-5 text-center">
            <div className="flex items-center gap-2 pt-2">
              <AriaAvatar size={40} />
              <p className="text-base font-semibold">Hi, I’m Aria.</p>
            </div>
            <div className="flex w-full flex-col gap-2" aria-label="Suggested questions">
              {SUGGESTED_QUESTIONS.map((question) => (
                <Button
                  key={question}
                  variant="outline"
                  className="h-auto min-h-10 justify-start whitespace-normal px-3 py-2 text-left font-normal"
                  disabled={questionsDisabled}
                  onClick={() => onQuestion(question)}
                >
                  {question}
                </Button>
              ))}
            </div>
            <p className="max-w-64 text-xs leading-relaxed text-muted-foreground">
              {readOnly ? "Answers use usage data your account can access." : "You review every change before it runs."}
            </p>
          </div>
        )}
        <MessageScroller.Button
          aria-label="Jump to latest"
          className="absolute bottom-3 left-1/2 -translate-x-1/2 rounded-full shadow-sm data-[active=false]:pointer-events-none data-[active=false]:opacity-0"
          render={<Button variant="outline" size="icon-sm" />}
        >
          <ArrowDown className="size-4" />
        </MessageScroller.Button>
      </MessageScroller.Root>
    </MessageScroller.Provider>
  );
}

function ActionCard({
  entry,
  open,
  reviewRef,
  onAnswer,
}: Pick<ConversationProps, "open" | "reviewRef" | "onAnswer"> & { entry: ActionEntry }) {
  const { scrollToEnd } = useMessageScroller();
  const review = entry.status === "review";
  useEffect(() => {
    if (!open || !review) return;
    scrollToEnd();
  }, [entry.id, open, review, scrollToEnd]);
  const answer = (approved: boolean) => {
    scrollToEnd();
    onAnswer(entry.id, approved);
  };
  const fields = (
    <div className="max-h-56 overflow-y-auto">
      <Table>
        <TableBody>
          {Object.entries(entry.action.arguments).map(([name, value]) => (
            <TableRow key={name}>
              <TableCell className="w-1/3 align-top text-muted-foreground whitespace-normal capitalize">
                {name.replaceAll("_", " ")}
              </TableCell>
              <TableCell className="whitespace-pre-wrap break-all">
                {typeof value === "string" ? value : JSON.stringify(value)}
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
  return (
    <Card
      ref={review ? reviewRef : undefined}
      tabIndex={review ? -1 : undefined}
      role="region"
      aria-label={entry.action.title}
      size="sm"
    >
      <CardHeader className="gap-2">
        <CardTitle>{entry.action.title}</CardTitle>
        <Badge variant={entry.status === "unknown" ? "destructive" : "secondary"} role="status">
          {entry.status === "completed" && <Check className="size-3" />}
          {ACTION_STATUS[entry.status]}
        </Badge>
      </CardHeader>
      <CardContent className="space-y-3">
        {review || entry.status === "applying" ? (
          fields
        ) : (
          <Collapsible>
            <CollapsibleTrigger render={<Button variant="ghost" size="xs" />}>
              <ChevronDown className="size-3" />
              Details
            </CollapsibleTrigger>
            <CollapsibleContent>{fields}</CollapsibleContent>
          </Collapsible>
        )}
        {entry.status === "unknown" && (
          <Alert variant="destructive">
            <AlertDescription>{entry.message}</AlertDescription>
          </Alert>
        )}
        {entry.status === "completed" && entry.key && (
          <div className="space-y-2">
            <div className="flex items-start gap-2">
              <code className="min-w-0 flex-1 break-all text-xs">{entry.key}</code>
              <CopyButton value={entry.key} label="Copy generated key" />
            </div>
            <p className="text-xs text-muted-foreground">Copy this key now. It stays only in this chat session.</p>
          </div>
        )}
      </CardContent>
      {review && (
        <CardFooter className="justify-end gap-2">
          <Button variant="outline" onClick={() => answer(false)}>
            Cancel
          </Button>
          <Button variant={entry.action.destructive ? "destructive" : "default"} onClick={() => answer(true)}>
            Confirm change
          </Button>
        </CardFooter>
      )}
    </Card>
  );
}
