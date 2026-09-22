"use client";

import { useEffect, useRef, useState } from "react";
import { MessageCircle, Plus, Send, Copy, Check } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { useAuth } from "@/contexts/AuthContext";
import { ApiError } from "@/lib/http/client";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Skeleton } from "@/components/ui/skeleton";
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle, SheetTrigger } from "@/components/ui/sheet";
import { getLiteAskSession } from "./session";
import { buildLiteAskHistory } from "./history";
import {
  liteAskApi,
  type JsonValue,
  type LiteAskConfig,
  type LiteAskMessage,
  type LiteAskProposal,
  type LiteAskResponse,
} from "./api";

interface DisplayMessage extends LiteAskMessage {
  result?: JsonValue;
}

function GeneratedKey({ value }: { value: string }) {
  const [copied, setCopied] = useState(false);
  const [copyFailed, setCopyFailed] = useState(false);
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(value);
      setCopied(true);
      setCopyFailed(false);
    } catch {
      setCopyFailed(true);
    }
  };
  return (
    <section aria-label="New API key" className="space-y-2 rounded-lg border bg-card p-4">
      <h3 className="text-sm font-medium">Your new API key</h3>
      <p className="text-xs text-muted-foreground">Copy it now. It is not saved in this conversation.</p>
      <code className="block break-all rounded-md bg-muted p-2 text-xs">{value}</code>
      <Button variant="outline" size="sm" onClick={copy}>
        {copied ? <Check /> : <Copy />}
        {copied ? "Copied" : "Copy key"}
      </Button>
      {copyFailed && (
        <p role="alert" className="text-xs text-destructive">
          Could not copy. Select and copy the key above.
        </p>
      )}
    </section>
  );
}

function Markdown({ content }: { content: string }) {
  return (
    <div className="space-y-2 break-words text-sm leading-relaxed [&_pre]:overflow-x-auto [&_pre]:rounded-md [&_pre]:bg-muted [&_pre]:p-3 [&_code]:font-mono [&_code]:text-xs [&_ul]:list-disc [&_ul]:pl-5 [&_ol]:list-decimal [&_ol]:pl-5">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        skipHtml
        components={{
          img: () => null,
          a: ({ children }) => <span>{children}</span>,
          table: ({ children }) => (
            <div className="overflow-x-auto">
              <table className="w-full text-xs [&_th]:border [&_th]:bg-muted [&_th]:p-2 [&_th]:text-left [&_td]:border [&_td]:p-2">
                {children}
              </table>
            </div>
          ),
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}

function Proposal({
  proposal,
  onApprove,
  onCancel,
}: {
  proposal: LiteAskProposal;
  onApprove: () => void;
  onCancel: () => void;
}) {
  const [expired, setExpired] = useState(() => proposal.expires_at * 1000 <= Date.now());
  useEffect(() => {
    const timeout = window.setTimeout(() => setExpired(true), Math.max(0, proposal.expires_at * 1000 - Date.now()));
    return () => window.clearTimeout(timeout);
  }, [proposal.expires_at]);
  return (
    <section aria-label="Review change" className="space-y-3 rounded-lg border bg-card p-4">
      <h3 className="font-medium">Review change</h3>
      <p className="text-sm">{proposal.title}</p>
      <code className="block text-xs text-muted-foreground">{proposal.tool}</code>
      <pre className="max-h-56 overflow-auto whitespace-pre-wrap break-all rounded-md bg-muted p-3 text-xs">
        {JSON.stringify(proposal.arguments, null, 2)}
      </pre>
      {expired && (
        <p role="status" className="text-xs text-muted-foreground">
          This approval expired. Ask for the change again.
        </p>
      )}
      <div className="flex gap-2">
        <Button onClick={onApprove} disabled={expired}>
          Confirm change
        </Button>
        <Button variant="outline" onClick={onCancel}>
          Cancel
        </Button>
      </div>
    </section>
  );
}

function LiteAskConversation({
  accessToken,
  config,
  onUnauthorized,
}: {
  accessToken: string;
  config: LiteAskConfig;
  onUnauthorized: () => void;
}) {
  const [open, setOpen] = useState(false);
  const [conversationId, setConversationId] = useState(() => crypto.randomUUID());
  const [messages, setMessages] = useState<DisplayMessage[]>([]);
  const [draft, setDraft] = useState("");
  const [requestKind, setRequestKind] = useState<"chat" | "approval" | null>(null);
  const busy = requestKind !== null;
  const [error, setError] = useState<string | null>(null);
  const [proposal, setProposal] = useState<LiteAskProposal | null>(null);
  const [generatedKey, setGeneratedKey] = useState<string | null>(null);
  const requestRef = useRef<AbortController | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => () => requestRef.current?.abort(), []);
  useEffect(() => {
    bottomRef.current?.scrollIntoView?.({ block: "end" });
  }, [messages, proposal, busy, generatedKey]);

  const receive = (response: LiteAskResponse) => {
    setMessages((previous) => [...previous, { role: "assistant", content: response.message, result: response.result }]);
    setProposal(response.proposal ?? null);
    setGeneratedKey(response.generated_key ?? null);
  };

  const fail = (failure: unknown, approval: boolean) => {
    if (failure instanceof ApiError && (failure.status === 401 || failure.status === 403)) {
      onUnauthorized();
      return;
    }
    setError(
      approval
        ? "Could not confirm the outcome. Check audit logs before trying this change again."
        : "LiteAsk could not answer. Try again in a moment.",
    );
  };

  const send = async () => {
    const content = draft.trim();
    const requestInProgress = busy || requestRef.current !== null;
    if (!content || proposal || requestInProgress) return;
    const controller = new AbortController();
    requestRef.current = controller;
    const history = buildLiteAskHistory([...messages, { role: "user", content }]);
    setMessages((previous) => [...previous, { role: "user", content }]);
    setDraft("");
    setRequestKind("chat");
    setError(null);
    setGeneratedKey(null);
    try {
      const response = await liteAskApi.chat(
        accessToken,
        { conversation_id: conversationId, messages: history },
        controller.signal,
      );
      if (!controller.signal.aborted) receive(response);
    } catch (failure) {
      if (!controller.signal.aborted) fail(failure, false);
    } finally {
      if (!controller.signal.aborted) {
        requestRef.current = null;
        setRequestKind(null);
      }
    }
  };

  const approve = async () => {
    if (!proposal || busy || requestRef.current) return;
    const controller = new AbortController();
    requestRef.current = controller;
    setProposal(null);
    setRequestKind("approval");
    setError(null);
    setGeneratedKey(null);
    try {
      const response = await liteAskApi.approve(
        accessToken,
        { conversation_id: conversationId, token: proposal.token },
        controller.signal,
      );
      if (!controller.signal.aborted) receive(response);
    } catch (failure) {
      if (!controller.signal.aborted) fail(failure, true);
    } finally {
      if (!controller.signal.aborted) {
        requestRef.current = null;
        setRequestKind(null);
      }
    }
  };

  const reset = () => {
    if (requestKind === "approval") return;
    requestRef.current?.abort();
    requestRef.current = null;
    setConversationId(crypto.randomUUID());
    setMessages([]);
    setDraft("");
    setProposal(null);
    setGeneratedKey(null);
    setError(null);
    setRequestKind(null);
  };

  const cancel = () => {
    setProposal(null);
    setMessages((previous) => [...previous, { role: "assistant", content: "Change canceled. Nothing was applied." }]);
  };

  return (
    <Sheet open={open} onOpenChange={setOpen}>
      <SheetTrigger
        render={<Button className="fixed bottom-5 right-5 z-overlay rounded-full px-4 shadow-lg" size="lg" />}
      >
        <MessageCircle aria-hidden="true" />
        LiteAsk
      </SheetTrigger>
      <SheetContent className="gap-0 data-[side=right]:w-full data-[side=right]:sm:w-[440px] data-[side=right]:sm:max-w-[440px]">
        <SheetHeader className="border-b pr-12">
          <SheetTitle className="text-base font-semibold tracking-tight">LiteAsk</SheetTitle>
          <SheetDescription>Ask about your gateway. Review changes before they run.</SheetDescription>
        </SheetHeader>
        <div className="flex items-center justify-between border-b px-4 py-2">
          <span className="truncate text-xs text-muted-foreground" title={config.model ?? undefined}>
            {config.model}
          </span>
          <Button variant="ghost" size="sm" onClick={reset} disabled={requestKind === "approval"}>
            <Plus />
            New chat
          </Button>
        </div>
        <div
          role="log"
          aria-label="LiteAsk conversation"
          aria-live="polite"
          className="min-h-0 flex-1 space-y-4 overflow-y-auto p-4"
        >
          {messages.length === 0 && (
            <div className="space-y-2 py-8">
              <h3 className="font-medium">How can I help?</h3>
              <p className="text-sm text-muted-foreground">
                Try “Show this month’s spend” or “Create a key for my team”.
              </p>
            </div>
          )}
          {messages.map((message, index) => (
            <div key={index} className={message.role === "user" ? "ml-8 rounded-2xl bg-muted px-4 py-3" : "space-y-2"}>
              <span className="sr-only">{message.role === "user" ? "You" : "LiteAsk"}</span>
              {message.role === "user" ? (
                <p className="whitespace-pre-wrap break-words text-sm">{message.content}</p>
              ) : (
                <Markdown content={message.content} />
              )}
              {message.result != null && (
                <details>
                  <summary className="cursor-pointer text-xs text-muted-foreground">View result</summary>
                  <pre className="mt-2 max-h-64 overflow-auto whitespace-pre-wrap break-all rounded-md bg-muted p-3 text-xs">
                    {JSON.stringify(message.result, null, 2)}
                  </pre>
                </details>
              )}
            </div>
          ))}
          {busy && (
            <div role="status" aria-label="LiteAsk is working" className="space-y-2">
              <Skeleton className="h-3 w-2/3" />
              <Skeleton className="h-3 w-1/2" />
              <span className="sr-only">LiteAsk is working</span>
            </div>
          )}
          {proposal && <Proposal key={proposal.token} proposal={proposal} onApprove={approve} onCancel={cancel} />}
          {generatedKey && <GeneratedKey key={generatedKey} value={generatedKey} />}
          {error && (
            <p role="alert" className="text-sm text-destructive">
              {error}
            </p>
          )}
          <div ref={bottomRef} />
        </div>
        <form
          className="space-y-2 border-t p-4"
          onSubmit={(event) => {
            event.preventDefault();
            void send();
          }}
        >
          {!config.can_execute_mutations && (
            <p className="text-xs text-muted-foreground">Read-only mode. Changes are currently unavailable.</p>
          )}
          <label htmlFor="liteask-message" className="sr-only">
            Message LiteAsk
          </label>
          <Textarea
            id="liteask-message"
            value={draft}
            onChange={(event) => setDraft(event.target.value)}
            placeholder="Ask LiteAsk…"
            maxLength={8000}
            rows={3}
            disabled={busy || Boolean(proposal)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
                event.preventDefault();
                void send();
              }
            }}
          />
          <div className="flex items-center justify-between gap-2">
            <span className="text-xs text-muted-foreground">
              {proposal ? "Confirm or cancel the change to continue" : "Enter to send · Shift+Enter for a new line"}
            </span>
            <Button type="submit" disabled={busy || Boolean(proposal) || !draft.trim()} aria-label="Send message">
              <Send aria-hidden="true" />
            </Button>
          </div>
        </form>
      </SheetContent>
    </Sheet>
  );
}

function LiteAskSession({ accessToken, expiresAt }: { accessToken: string; expiresAt: number | null }) {
  const [config, setConfig] = useState<LiteAskConfig | null>(null);
  const [expired, setExpired] = useState(false);
  useEffect(() => {
    const controller = new AbortController();
    liteAskApi
      .config(accessToken, controller.signal)
      .then((value) => {
        if (!controller.signal.aborted) setConfig(value);
      })
      .catch(() => {});
    return () => controller.abort();
  }, [accessToken]);
  useEffect(() => {
    if (expiresAt === null) return;
    const timeout = window.setTimeout(() => setExpired(true), Math.max(0, expiresAt - Date.now()));
    return () => window.clearTimeout(timeout);
  }, [expiresAt]);
  if (expired || !config?.enabled || !config.model) return null;
  return <LiteAskConversation accessToken={accessToken} config={config} onUnauthorized={() => setConfig(null)} />;
}

export default function LiteAskWidget() {
  const auth = useAuth();
  const session = getLiteAskSession(auth);
  if (!session || !auth.accessToken) return null;
  return (
    <LiteAskSession
      key={`${auth.token}:${auth.accessToken}:${auth.userID}`}
      accessToken={auth.accessToken}
      expiresAt={session.expiresAt}
    />
  );
}
