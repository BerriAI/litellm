import { useEffect, type RefObject } from "react";
import type { ParseKeys } from "i18next";
import { ArrowDown, Check, ChevronDown } from "lucide-react";
import { useTranslation } from "react-i18next";
import { MessageScroller, useMessageScroller } from "@shadcn/react/message-scroller";
import CopyButton from "@/components/shared/CopyButton";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { Skeleton } from "@/components/ui/skeleton";
import { Table, TableBody, TableCell, TableRow } from "@/components/ui/table";
import type { ActionEntry, ConversationEntry } from "./useLiteAdmin";
import { AriaAnswer } from "./AriaAnswer";
import { ARIA_AGENT_LABEL } from "./aria-brand";

const ACTION_STATUS: Record<ActionEntry["status"], ParseKeys> = {
  review: "aria.conversation.status.review",
  applying: "aria.conversation.status.applying",
  completed: "aria.conversation.status.completed",
  cancelled: "aria.conversation.status.cancelled",
  unknown: "aria.conversation.status.unknown",
};

const ACTION_TITLES: Record<string, ParseKeys> = {
  key_create: "aria.actions.keyCreate",
  key_update: "aria.actions.keyUpdate",
  key_detach_from_team: "aria.actions.keyDetachFromTeam",
  key_delete: "aria.actions.keyDelete",
  key_block: "aria.actions.keyBlock",
  key_unblock: "aria.actions.keyUnblock",
  team_create: "aria.actions.teamCreate",
  team_update: "aria.actions.teamUpdate",
  team_delete: "aria.actions.teamDelete",
  team_member_add: "aria.actions.teamMemberAdd",
  team_member_update: "aria.actions.teamMemberUpdate",
  team_member_delete: "aria.actions.teamMemberDelete",
  user_create: "aria.actions.userCreate",
  user_update: "aria.actions.userUpdate",
  user_delete: "aria.actions.userDelete",
  budget_create: "aria.actions.budgetCreate",
  budget_update: "aria.actions.budgetUpdate",
  budget_delete: "aria.actions.budgetDelete",
};

interface ConversationProps {
  entries: ConversationEntry[];
  thinking: boolean;
  open: boolean;
  reviewRef: RefObject<HTMLDivElement | null>;
  onAnswer: (id: string, approved: boolean) => void;
}

export function LiteAdminConversation({ entries, thinking, open, reviewRef, onAnswer }: ConversationProps) {
  const { t } = useTranslation();
  return (
    <MessageScroller.Provider autoScroll>
      <MessageScroller.Root className="relative flex min-h-0 flex-1 flex-col overflow-hidden">
        <MessageScroller.Viewport
          className="min-h-0 flex-1 overflow-y-auto overscroll-contain data-pending-scroll:invisible"
          aria-label={t("aria.conversation.ariaLabel", { agent: ARIA_AGENT_LABEL })}
        >
          <MessageScroller.Content className="flex flex-col gap-6 p-4 sm:p-6" aria-busy={thinking}>
            {entries.map((entry) => (
              <MessageScroller.Item key={entry.id} messageId={entry.id}>
                {entry.kind === "message" &&
                  (entry.message.role === "assistant" ? (
                    <AriaAnswer message={entry.message} chart={entry.chart} question={entry.question} />
                  ) : (
                    <article
                      aria-label={t("aria.conversation.yourQuestion")}
                      className="ml-auto w-fit max-w-[90%] whitespace-pre-wrap break-words rounded-xl bg-muted px-4 py-3 text-sm leading-relaxed"
                    >
                      {entry.message.content}
                    </article>
                  ))}
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
                <div
                  role="status"
                  aria-label={t("aria.conversation.workingLabel", { agent: ARIA_AGENT_LABEL })}
                  className="space-y-2"
                >
                  <Skeleton className="h-3 w-3/4" />
                  <Skeleton className="h-3 w-1/2" />
                  <span className="sr-only">{t("aria.conversation.working")}</span>
                </div>
              </MessageScroller.Item>
            )}
          </MessageScroller.Content>
        </MessageScroller.Viewport>
        <MessageScroller.Button
          aria-label={t("aria.conversation.jumpToLatest")}
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
  const { t } = useTranslation();
  const { scrollToEnd } = useMessageScroller();
  const review = entry.status === "review";
  const title = Object.hasOwn(ACTION_TITLES, entry.action.name)
    ? t(ACTION_TITLES[entry.action.name])
    : entry.action.title;
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
      aria-label={title}
      size="sm"
    >
      <CardHeader className="gap-2">
        <CardTitle>{title}</CardTitle>
        <Badge variant={entry.status === "unknown" ? "destructive" : "secondary"} role="status">
          {entry.status === "completed" && <Check className="size-3" />}
          {t(ACTION_STATUS[entry.status])}
        </Badge>
      </CardHeader>
      <CardContent className="space-y-3">
        {review || entry.status === "applying" ? (
          fields
        ) : (
          <Collapsible>
            <CollapsibleTrigger render={<Button variant="ghost" size="xs" />}>
              <ChevronDown className="size-3" />
              {t("aria.conversation.details")}
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
              <CopyButton value={entry.key} label={t("aria.conversation.copyGeneratedKey")} />
            </div>
            <p className="text-xs text-muted-foreground">{t("aria.conversation.keyNotice")}</p>
          </div>
        )}
      </CardContent>
      {review && (
        <CardFooter className="justify-end gap-2">
          <Button variant="outline" onClick={() => answer(false)}>
            {t("aria.conversation.cancel")}
          </Button>
          <Button variant={entry.action.destructive ? "destructive" : "default"} onClick={() => answer(true)}>
            {t("aria.conversation.confirmChange")}
          </Button>
        </CardFooter>
      )}
    </Card>
  );
}
