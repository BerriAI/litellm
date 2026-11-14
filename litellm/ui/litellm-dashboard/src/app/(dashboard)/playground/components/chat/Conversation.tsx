"use client";

import type { ReactNode } from "react";
import {
  ArrowDownToLine,
  ArrowUpFromLine,
  Brain,
  Clock,
  Coins,
  DatabaseZap,
  FileText,
  Music,
  PlayCircle,
  Repeat,
  ScanSearch,
  Timer,
  Wrench,
} from "lucide-react";
import type { TFunction } from "i18next";
import { Trans, useTranslation } from "react-i18next";
import i18n from "@/i18n/i18n";
import { EndpointType } from "@/components/chat_ui/mode_endpoint_mapping";
import MCPEventsDisplay from "@/components/chat_ui/MCPEventsDisplay";
import ReasoningContent from "@/components/chat_ui/ReasoningContent";
import type { MessageAttachment, MessageType } from "@/components/chat_ui/types";
import type { CodeInterpreterResult } from "@/components/llm_calls/code_interpreter_handler";
import type { MCPEvent } from "@/components/mcp_tools/types";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import {
  Attachment,
  AttachmentContent,
  AttachmentDescription,
  AttachmentGroup,
  AttachmentMedia,
  AttachmentTitle,
} from "@/components/ui/attachment";
import { Badge } from "@/components/ui/badge";
import { Bubble, BubbleContent } from "@/components/ui/Bubble";
import { Button } from "@/components/ui/button";
import { Empty, EmptyContent, EmptyDescription, EmptyHeader, EmptyMedia, EmptyTitle } from "@/components/ui/empty";
import { Marker, MarkerContent, MarkerIcon } from "@/components/ui/marker";
import { Message, MessageContent, MessageFooter, MessageHeader } from "@/components/ui/message";
import {
  MessageScroller,
  MessageScrollerButton,
  MessageScrollerContent,
  MessageScrollerItem,
  MessageScrollerProvider,
  MessageScrollerViewport,
} from "@/components/ui/message-scroller";
import { Spinner } from "@/components/ui/spinner";
import { Toggle } from "@/components/ui/toggle";
import A2AMetrics from "../chat_ui/A2AMetrics";
import CodeInterpreterOutput from "../chat_ui/CodeInterpreterOutput";
import { SearchResultsDisplay } from "../chat_ui/SearchResultsDisplay";
import { formatBytes } from "./composerFiles";
import { CopyIconButton, FallbackBadge, LongValue } from "./ModelName";
import { ReplyBody } from "./ReplyContent";

export const formatSeconds = (ms: number): string => `${(ms / 1000).toFixed(2)}s`;

export const formatCost = (cost: number): string =>
  `$${cost < 0.0001 && cost > 0 ? cost.toPrecision(2) : cost.toFixed(4)}`;

const MCP_EVENT_ENDPOINTS: readonly EndpointType[] = [EndpointType.CHAT, EndpointType.RESPONSES];

const textOf = (message: MessageType): string => (typeof message.content === "string" ? message.content : "");

const fellBack = (message: MessageType): boolean => (message.trace?.attemptedFallbacks ?? 0) > 0;

export const servedModel = (message: MessageType): string | undefined =>
  message.trace?.modelGroup ?? (message.isError ? undefined : message.model ?? message.trace?.requestedModel);

const replyModel = (
  message: MessageType,
  t: TFunction = i18n.t.bind(i18n),
): Readonly<{ value: string; detail: string }> | null => {
  const served = servedModel(message);
  if (served)
    return {
      value: served,
      detail: fellBack(message)
        ? t("playground.chat.conversation.servedByAfterFallback")
        : t("playground.chat.conversation.servedBy"),
    };
  const requested = message.trace?.requestedModel ?? message.model;
  return requested ? { value: requested, detail: t("playground.chat.conversation.requestedModel") } : null;
};

function Metric({ icon, value, label }: { icon: ReactNode; value: string; label?: string }) {
  return (
    <span className="inline-flex items-center gap-1 whitespace-nowrap [&_svg]:size-3.5">
      {icon}
      <span className="text-foreground">{value}</span>
      {label}
    </span>
  );
}

function ReplyMetrics({ message }: { message: MessageType }) {
  const { t } = useTranslation();
  const usage = message.usage;
  const cost = usage?.cost ?? message.trace?.cost;
  const cached = usage?.servedFromResponseCache || message.trace?.cacheHit;
  return (
    <div
      aria-label={t("playground.chat.conversation.metrics.label")}
      className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-muted-foreground tabular-nums"
    >
      {message.timeToFirstToken !== undefined && (
        <Metric
          icon={<Timer />}
          value={formatSeconds(message.timeToFirstToken)}
          label={t("playground.chat.conversation.metrics.firstToken")}
        />
      )}
      {message.totalLatency !== undefined && (
        <Metric
          icon={<Clock />}
          value={formatSeconds(message.totalLatency)}
          label={t("playground.chat.conversation.metrics.total")}
        />
      )}
      {usage?.promptTokens !== undefined && (
        <Metric
          icon={<ArrowDownToLine />}
          value={String(usage.promptTokens)}
          label={t("playground.chat.conversation.metrics.in")}
        />
      )}
      {usage?.completionTokens !== undefined && (
        <Metric
          icon={<ArrowUpFromLine />}
          value={String(usage.completionTokens)}
          label={t("playground.chat.conversation.metrics.out")}
        />
      )}
      {usage?.reasoningTokens ? (
        <Metric
          icon={<Brain />}
          value={String(usage.reasoningTokens)}
          label={t("playground.chat.conversation.metrics.reasoning")}
        />
      ) : null}
      {usage?.cacheReadTokens ? (
        <Metric
          icon={<DatabaseZap />}
          value={String(usage.cacheReadTokens)}
          label={t("playground.chat.conversation.metrics.cached")}
        />
      ) : null}
      {cost !== undefined && <Metric icon={<Coins />} value={formatCost(cost)} />}
      {message.toolName && <Metric icon={<Wrench />} value={message.toolName} />}
      {cached && <Badge variant="secondary">{t("playground.chat.conversation.metrics.cachedResponse")}</Badge>}
    </div>
  );
}

function AttachmentMediaFor({ attachment }: { attachment: MessageAttachment }) {
  if (attachment.previewUrl)
    return (
      <AttachmentMedia variant="image">
        <img src={attachment.previewUrl} alt="" />
      </AttachmentMedia>
    );
  return <AttachmentMedia>{attachment.kind === "audio" ? <Music /> : <FileText />}</AttachmentMedia>;
}

function UserTurn({ message }: { message: MessageType }) {
  const text = textOf(message);
  return (
    <Message align="end">
      <MessageContent>
        {message.attachments && message.attachments.length > 0 && (
          <AttachmentGroup className="justify-end">
            {message.attachments.map((attachment, index) => (
              <Attachment key={`${attachment.name}-${index}`} size="sm">
                <AttachmentMediaFor attachment={attachment} />
                <AttachmentContent>
                  <AttachmentTitle>{attachment.name}</AttachmentTitle>
                  <AttachmentDescription>{formatBytes(attachment.size)}</AttachmentDescription>
                </AttachmentContent>
              </Attachment>
            ))}
          </AttachmentGroup>
        )}
        {message.mcpToolCall && (
          <Bubble variant="muted" align="end">
            <BubbleContent className="flex flex-col gap-1">
              <span className="flex items-center gap-1.5 text-xs text-muted-foreground">
                <Wrench className="size-3.5" />
                <Trans
                  i18nKey="playground.chat.conversation.calledTool"
                  components={{ tool: <span className="font-mono text-foreground">{message.mcpToolCall.tool}</span> }}
                />
              </span>
              <pre className="font-mono text-xs whitespace-pre-wrap">
                {JSON.stringify(message.mcpToolCall.arguments, null, 2)}
              </pre>
            </BubbleContent>
          </Bubble>
        )}
        {text && (
          <Bubble variant="muted" align="end">
            <BubbleContent className="whitespace-pre-wrap">{text}</BubbleContent>
          </Bubble>
        )}
      </MessageContent>
    </Message>
  );
}

export type LastReplyExtras = Readonly<{
  endpoint: EndpointType;
  mcpEvents: readonly MCPEvent[];
  codeInterpreter: CodeInterpreterResult | null;
  accessToken: string;
}>;

function AssistantReply({
  message,
  isLast,
  extras,
  inspected,
  onInspect,
}: {
  message: MessageType;
  isLast: boolean;
  extras: LastReplyExtras;
  inspected: boolean;
  onInspect: () => void;
}) {
  const { t } = useTranslation();
  const model = replyModel(message, t);
  const showMcpEvents = isLast && extras.mcpEvents.length > 0 && MCP_EVENT_ENDPOINTS.includes(extras.endpoint);
  const showCode = isLast && extras.codeInterpreter && extras.endpoint === EndpointType.RESPONSES;
  return (
    <Message>
      <MessageContent className="text-sm">
        {model && (
          <MessageHeader className="gap-2 px-0 font-normal">
            <LongValue
              value={model.value}
              detail={model.detail}
              copyLabel={t("playground.chat.conversation.modelNameCopy")}
              className="max-w-96 text-xs text-muted-foreground"
            />
            {fellBack(message) && <FallbackBadge />}
          </MessageHeader>
        )}
        {message.reasoningContent && <ReasoningContent reasoningContent={message.reasoningContent} />}
        {showMcpEvents && <MCPEventsDisplay events={[...extras.mcpEvents]} />}
        {message.searchResults && <SearchResultsDisplay searchResults={message.searchResults} />}
        {showCode && extras.codeInterpreter && (
          <CodeInterpreterOutput
            code={extras.codeInterpreter.code}
            containerId={extras.codeInterpreter.containerId}
            annotations={extras.codeInterpreter.annotations}
            accessToken={extras.accessToken}
          />
        )}
        {message.isError ? (
          <Alert variant="destructive">
            <AlertTitle>{t("playground.chat.conversation.requestFailed")}</AlertTitle>
            <AlertDescription className="font-mono text-xs wrap-anywhere">{textOf(message)}</AlertDescription>
          </Alert>
        ) : (
          <ReplyBody message={message} />
        )}
        <MessageFooter className="items-center gap-2 px-0 font-normal">
          <div className="min-w-0 flex-1">
            {message.a2aMetadata ? (
              <A2AMetrics
                a2aMetadata={message.a2aMetadata}
                timeToFirstToken={message.timeToFirstToken}
                totalLatency={message.totalLatency}
              />
            ) : (
              <ReplyMetrics message={message} />
            )}
          </div>
          <span className="flex shrink-0 items-center gap-0.5">
            {!message.isImage && !message.isAudio && (
              <CopyIconButton value={textOf(message)} label={t("playground.chat.conversation.replyCopy")} />
            )}
            {message.trace && (
              <Toggle
                size="sm"
                pressed={inspected}
                onPressedChange={onInspect}
                aria-label={t("playground.chat.conversation.inspectLabel")}
                className="h-6 gap-1 px-1.5 text-xs"
              >
                <ScanSearch />
                {t("playground.chat.conversation.inspect")}
              </Toggle>
            )}
          </span>
        </MessageFooter>
      </MessageContent>
    </Message>
  );
}

function WaitingTurn({ target }: { target: string | undefined }) {
  const { t } = useTranslation();
  return (
    <Message>
      <MessageContent className="text-sm">
        <div className="flex items-center gap-2 text-muted-foreground">
          <Spinner className="size-3.5" />
          <span className="shimmer">
            {target
              ? t("playground.chat.conversation.waitingFor", { target })
              : t("playground.chat.conversation.waitingForReply")}
          </span>
        </div>
      </MessageContent>
    </Message>
  );
}

function SwitchMarker({ model }: { model: string }) {
  const { t } = useTranslation();
  return (
    <Marker variant="separator">
      <MarkerIcon>
        <Repeat />
      </MarkerIcon>
      <MarkerContent>{t("playground.chat.conversation.switchedTo", { model })}</MarkerContent>
    </Marker>
  );
}

function EmptyConversation({
  suggestions,
  onPick,
}: {
  suggestions: readonly string[];
  onPick: (text: string) => void;
}) {
  const { t } = useTranslation();
  return (
    <Empty className="flex-1">
      <EmptyHeader>
        <EmptyMedia variant="icon">
          <PlayCircle />
        </EmptyMedia>
        <EmptyTitle>{t("playground.chat.conversation.empty.title")}</EmptyTitle>
        <EmptyDescription>{t("playground.chat.conversation.empty.description")}</EmptyDescription>
      </EmptyHeader>
      {suggestions.length > 0 && (
        <EmptyContent className="max-w-xl flex-row flex-wrap justify-center gap-2">
          {suggestions.map((suggestion) => (
            <Button key={suggestion} variant="outline" size="sm" onClick={() => onPick(suggestion)}>
              {suggestion}
            </Button>
          ))}
        </EmptyContent>
      )}
    </Empty>
  );
}

const switchedModel = (messages: readonly MessageType[], index: number): string | null => {
  const message = messages[index];
  if (message?.role !== "user" || !message.model) return null;
  const previous = messages.slice(0, index).findLast((candidate) => candidate.role === "user");
  return previous?.model && previous.model !== message.model ? message.model : null;
};

export function Conversation({
  messages,
  isLoading,
  waitingFor,
  extras,
  inspectedIndex,
  onInspect,
  suggestions,
  onPickSuggestion,
}: {
  messages: readonly MessageType[];
  isLoading: boolean;
  waitingFor: string | undefined;
  extras: LastReplyExtras;
  inspectedIndex: number | null;
  onInspect: (index: number) => void;
  suggestions: readonly string[];
  onPickSuggestion: (text: string) => void;
}) {
  const { t } = useTranslation();
  if (messages.length === 0) return <EmptyConversation suggestions={suggestions} onPick={onPickSuggestion} />;
  const lastIndex = messages.length - 1;
  const waiting = isLoading && messages[lastIndex]?.role === "user";
  return (
    <MessageScrollerProvider autoScroll defaultScrollPosition="end">
      <MessageScroller className="min-h-0 flex-1">
        <MessageScrollerViewport aria-label={t("playground.chat.conversation.label")} className="scroll-fade-b">
          <MessageScrollerContent className="mx-auto w-full max-w-3xl gap-6 px-6 py-6">
            {messages.flatMap((message, index) => {
              const switched = switchedModel(messages, index);
              const turn = (
                <MessageScrollerItem key={`m${index}`} messageId={`m${index}`} scrollAnchor={index === lastIndex}>
                  {message.role === "user" ? (
                    <UserTurn message={message} />
                  ) : (
                    <AssistantReply
                      message={message}
                      isLast={index === lastIndex}
                      extras={extras}
                      inspected={inspectedIndex === index}
                      onInspect={() => onInspect(index)}
                    />
                  )}
                </MessageScrollerItem>
              );
              return switched
                ? [
                    <MessageScrollerItem key={`s${index}`} messageId={`s${index}`}>
                      <SwitchMarker model={switched} />
                    </MessageScrollerItem>,
                    turn,
                  ]
                : [turn];
            })}
            {waiting && (
              <MessageScrollerItem key="waiting" messageId="waiting" scrollAnchor>
                <WaitingTurn target={waitingFor} />
              </MessageScrollerItem>
            )}
          </MessageScrollerContent>
        </MessageScrollerViewport>
        <MessageScrollerButton />
      </MessageScroller>
    </MessageScrollerProvider>
  );
}
