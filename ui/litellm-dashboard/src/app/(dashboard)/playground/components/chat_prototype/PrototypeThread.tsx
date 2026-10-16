"use client";

import { ArrowUp, FileText, MessageSquare, Paperclip, Repeat, X } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import ResponseMetrics from "@/components/chat_ui/ResponseMetrics";
import {
  Attachment,
  AttachmentAction,
  AttachmentActions,
  AttachmentContent,
  AttachmentDescription,
  AttachmentMedia,
  AttachmentTitle,
} from "@/components/ui/attachment";
import { Bubble, BubbleContent } from "@/components/ui/Bubble";
import { Button } from "@/components/ui/button";
import { Empty, EmptyContent, EmptyDescription, EmptyHeader, EmptyMedia, EmptyTitle } from "@/components/ui/empty";
import { InputGroup, InputGroupAddon, InputGroupButton, InputGroupTextarea } from "@/components/ui/input-group";
import { Kbd } from "@/components/ui/kbd";
import { Marker, MarkerContent, MarkerIcon } from "@/components/ui/marker";
import { Message, MessageContent, MessageFooter } from "@/components/ui/message";
import {
  MessageScroller,
  MessageScrollerButton,
  MessageScrollerContent,
  MessageScrollerItem,
  MessageScrollerProvider,
  MessageScrollerViewport,
} from "@/components/ui/message-scroller";
import { Spinner } from "@/components/ui/spinner";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";

export const PROTOTYPE_MODEL = "gpt-4o-mini";

export const TABLE_REPLY = `Here are three kinds of gateway that sit between apps and model providers:

| Gateway | Description |
| --- | --- |
| Self-hosted proxy | Runs in your own cloud and exposes one OpenAI-style API over many providers |
| Managed gateway | A hosted service that adds logging, caching and fallbacks |
| API gateway plugin | Adds model routing to an API gateway you already run |

Pick the self-hosted proxy when requests must stay inside your network.`;

export const SUMMARY_REPLY = `The spec asks for three things:

1. Rate limits per team, enforced before the request reaches the provider
2. A fallback to a second deployment when the first returns a 429
3. Spend logged against the virtual key, not the team

Point 2 is already covered by router fallbacks. Points 1 and 3 need a team budget and key-level spend tracking.`;

export function Markdown({ children }: { children: string }) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      components={{
        p: ({ node: _node, ...props }) => <p className="my-2 first:mt-0 last:mb-0" {...props} />,
        ol: ({ node: _node, ...props }) => <ol className="my-2 list-decimal space-y-1 pl-5" {...props} />,
        table: ({ node: _node, ...props }) => (
          <div className="my-2 rounded-lg border border-border">
            <Table {...props} />
          </div>
        ),
        thead: ({ node: _node, ...props }) => <TableHeader {...props} />,
        tbody: ({ node: _node, ...props }) => <TableBody {...props} />,
        tr: ({ node: _node, ...props }) => <TableRow {...props} />,
        th: ({ node: _node, ...props }) => <TableHead {...props} />,
        td: ({ node: _node, ...props }) => <TableCell className="whitespace-normal" {...props} />,
      }}
    >
      {children}
    </ReactMarkdown>
  );
}

export function UserTurn({ text, file }: { text: string; file?: string }) {
  return (
    <Message align="end">
      <MessageContent>
        {file && (
          <Attachment size="sm">
            <AttachmentMedia>
              <FileText />
            </AttachmentMedia>
            <AttachmentContent>
              <AttachmentTitle>{file}</AttachmentTitle>
              <AttachmentDescription>PDF, 184 KB</AttachmentDescription>
            </AttachmentContent>
          </Attachment>
        )}
        <Bubble variant="muted" align="end">
          <BubbleContent>{text}</BubbleContent>
        </Bubble>
      </MessageContent>
    </Message>
  );
}

function AssistantTurn({ text, latency, cost }: { text: string; latency: number; cost: number }) {
  return (
    <Message>
      <MessageContent className="text-sm">
        <Markdown>{text}</Markdown>
        <MessageFooter className="px-0 font-normal">
          <ResponseMetrics
            timeToFirstToken={latency * 220}
            totalLatency={latency * 1000}
            usage={{ promptTokens: 38, completionTokens: 112, totalTokens: 150, cost }}
          />
        </MessageFooter>
      </MessageContent>
    </Message>
  );
}

export function StreamingTurn() {
  return (
    <Message>
      <MessageContent className="text-sm">
        <div className="flex items-center gap-2 text-muted-foreground">
          <Spinner className="size-3.5" />
          <span className="shimmer">Waiting for anthropic-sonnet-5</span>
        </div>
      </MessageContent>
    </Message>
  );
}

export function SwitchMarker({ model }: { model: string }) {
  return (
    <Marker variant="separator">
      <MarkerIcon>
        <Repeat />
      </MarkerIcon>
      <MarkerContent>Switched to {model}</MarkerContent>
    </Marker>
  );
}

function EmptyThread({ onPick }: { onPick?: (prompt: string) => void }) {
  return (
    <Empty className="flex-1">
      <EmptyHeader>
        <EmptyMedia variant="icon">
          <MessageSquare />
        </EmptyMedia>
        <EmptyTitle>Send a test request</EmptyTitle>
        <EmptyDescription>
          Requests go through this gateway with the model, key and guardrails set above, and show latency, tokens and
          cost on every reply
        </EmptyDescription>
      </EmptyHeader>
      <EmptyContent className="flex-row flex-wrap justify-center gap-2">
        {["Explain quantum computing", "Write a haiku about rate limits", "Summarize this PDF"].map((prompt) => (
          <Button key={prompt} variant="outline" size="sm" onClick={() => onPick?.(prompt)}>
            {prompt}
          </Button>
        ))}
      </EmptyContent>
    </Empty>
  );
}

function Composer({ withFile }: { withFile: boolean }) {
  return (
    <InputGroup>
      {withFile && (
        <InputGroupAddon align="block-start">
          <Attachment size="xs" state="uploading">
            <AttachmentMedia>
              <FileText />
            </AttachmentMedia>
            <AttachmentContent>
              <AttachmentTitle>team-budgets-spec.pdf</AttachmentTitle>
            </AttachmentContent>
            <AttachmentActions>
              <AttachmentAction aria-label="Remove file">
                <X />
              </AttachmentAction>
            </AttachmentActions>
          </Attachment>
        </InputGroupAddon>
      )}
      <InputGroupTextarea placeholder={`Message ${PROTOTYPE_MODEL}`} rows={2} aria-label="Message" />
      <InputGroupAddon align="block-end">
        <InputGroupButton size="icon-xs" aria-label="Attach file">
          <Paperclip />
        </InputGroupButton>
        <span className="ml-auto flex items-center gap-1 text-xs font-normal text-muted-foreground">
          <Kbd>Enter</Kbd> to send, <Kbd>Shift</Kbd>+<Kbd>Enter</Kbd> for a new line
        </span>
        <InputGroupButton size="icon-xs" variant="default" aria-label="Send">
          <ArrowUp />
        </InputGroupButton>
      </InputGroupAddon>
    </InputGroup>
  );
}

export type ThreadState = "filled" | "streaming" | "empty";

export function PrototypeThread({ state = "filled" }: { state?: ThreadState }) {
  return (
    <div className="flex min-h-0 flex-1 flex-col">
      {state === "empty" ? (
        <EmptyThread />
      ) : (
        <MessageScrollerProvider autoScroll defaultScrollPosition="end">
          <MessageScroller className="min-h-0 flex-1">
            <MessageScrollerViewport aria-label="Conversation" className="scroll-fade-b">
              <MessageScrollerContent className="mx-auto w-full max-w-3xl gap-6 px-6 py-6">
                <MessageScrollerItem messageId="u1">
                  <UserTurn text="Give me a 3-row markdown table of LLM gateways and one sentence on each" />
                </MessageScrollerItem>
                <MessageScrollerItem messageId="a1">
                  <AssistantTurn text={TABLE_REPLY} latency={1.42} cost={0.0001} />
                </MessageScrollerItem>
                <MessageScrollerItem messageId="m1">
                  <SwitchMarker model="anthropic-sonnet-5" />
                </MessageScrollerItem>
                <MessageScrollerItem messageId="u2">
                  <UserTurn text="What does this spec ask the gateway to do?" file="team-budgets-spec.pdf" />
                </MessageScrollerItem>
                <MessageScrollerItem messageId="a2" scrollAnchor={state !== "streaming"}>
                  {state === "streaming" ? (
                    <StreamingTurn />
                  ) : (
                    <AssistantTurn text={SUMMARY_REPLY} latency={2.8} cost={0.0042} />
                  )}
                </MessageScrollerItem>
              </MessageScrollerContent>
            </MessageScrollerViewport>
            <MessageScrollerButton />
          </MessageScroller>
        </MessageScrollerProvider>
      )}
      <div className="mx-auto w-full max-w-3xl px-6 pb-6">
        <Composer withFile={state === "streaming"} />
      </div>
    </div>
  );
}
