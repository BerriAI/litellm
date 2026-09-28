"use client";

import { useRef, useState, type ReactNode } from "react";
import { useQuery } from "@tanstack/react-query";
import { RotateCcw, Sparkles, X } from "lucide-react";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { useDisableLiteAdmin } from "@/app/(dashboard)/hooks/useDisableLiteAdmin";
import { useProxySettingsQuery } from "@/app/(dashboard)/hooks/proxySettings/useProxySettings";
import { ChatComposer } from "@/app/(dashboard)/playground/components/chat_ui/ChatComposer";
import { EndpointType, isModeCompatibleWithEndpoint } from "@/components/chat_ui/mode_endpoint_mapping";
import { fetchAvailableModels, type ModelGroup } from "@/components/llm_calls/fetch_models";
import { getProxyBaseUrl } from "@/components/networking";
import { SearchSelect } from "@/components/shared/SearchSelect";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { FieldError } from "@/components/ui/field";
import {
  Popover,
  PopoverContent,
  PopoverDescription,
  PopoverHeader,
  PopoverTitle,
  PopoverTrigger,
} from "@/components/ui/popover";
import { Skeleton } from "@/components/ui/skeleton";
import { isProxyAdminRole } from "@/utils/roles";
import { MAX_INPUT_LENGTH, resolveInferenceTarget } from "./agent";
import { LiteAdminConversation } from "./LiteAdminConversation";
import { useLiteAdmin, type LiteAdminSession } from "./useLiteAdmin";

const PANEL_CLASS =
  "flex h-[min(42rem,calc(100dvh-6rem))] w-[min(30rem,calc(100vw-2rem))] min-w-0 flex-col gap-0 overflow-hidden rounded-xl p-0";
type ManagementSession = Omit<LiteAdminSession, "inferenceBaseUrl">;

export default function LiteAdmin() {
  const auth = useAuthorized();
  const [disabled] = useDisableLiteAdmin(auth.userId);
  const sessionReady = !auth.isLoading && auth.isAuthorized;
  const writableAdmin = !auth.isViewOnly && isProxyAdminRole(auth.userRole);
  const allowed = sessionReady && writableAdmin && !disabled;
  if (!allowed || !auth.token || !auth.accessToken) return null;
  const session = { token: auth.token, accessToken: auth.accessToken, managementBaseUrl: getProxyBaseUrl() };
  return (
    <ConfiguredLiteAdmin
      key={JSON.stringify([auth.userId, session.token, session.accessToken, session.managementBaseUrl])}
      session={session}
    />
  );
}

function ConfiguredLiteAdmin({ session }: { session: ManagementSession }) {
  const [open, setOpen] = useState(false);
  const settings = useProxySettingsQuery(session.accessToken);
  const candidate =
    settings.data?.LITELLM_UI_API_DOC_BASE_URL?.trim() ||
    settings.data?.PROXY_BASE_URL?.trim() ||
    session.managementBaseUrl;
  const target = settings.data
    ? resolveInferenceTarget(candidate, session.managementBaseUrl, window.location.href)
    : null;
  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger render={<Button className="fixed right-5 bottom-5 z-floating rounded-full shadow-lg" />}>
        <Sparkles className="size-4" />
        LiteAdmin
      </PopoverTrigger>
      <Destination
        key={target?.baseUrl ?? "unavailable"}
        session={session}
        target={target}
        loading={settings.isPending}
        retry={() => void settings.refetch()}
        open={open}
        close={() => setOpen(false)}
      />
    </Popover>
  );
}

function Destination({
  session,
  target,
  loading,
  retry,
  open,
  close,
}: {
  session: ManagementSession;
  target: ReturnType<typeof resolveInferenceTarget> | null;
  loading: boolean;
  retry: () => void;
  open: boolean;
  close: () => void;
}) {
  const [approved, setApproved] = useState(false);
  if (target?.baseUrl && (!target.requiresConsent || approved)) {
    return <LiteAdminChat session={{ ...session, inferenceBaseUrl: target.baseUrl }} open={open} close={close} />;
  }
  return (
    <PopoverContent side="top" align="end" sideOffset={12} className={PANEL_CLASS}>
      <PanelHeader close={close} />
      <div className="p-4">
        {loading && <Skeleton className="h-24" aria-label="Loading gateway settings" />}
        {!loading && !target?.baseUrl && (
          <Alert variant="destructive">
            <AlertDescription>{target?.error || "Could not load gateway settings."}</AlertDescription>
            <Button variant="link" onClick={retry}>
              Retry
            </Button>
          </Alert>
        )}
        {!loading && target?.baseUrl && (
          <Card size="sm">
            <CardHeader>
              <CardTitle>Connect to the configured gateway</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <p className="break-all font-mono text-xs">{target.baseUrl}</p>
              <p className="text-muted-foreground">
                This gateway uses a different address. LiteAdmin will send your existing session credential to the
                address shown above.
              </p>
            </CardContent>
            <CardFooter>
              <Button onClick={() => setApproved(true)}>Use configured gateway</Button>
            </CardFooter>
          </Card>
        )}
      </div>
    </PopoverContent>
  );
}

function LiteAdminChat({ session, open, close }: { session: LiteAdminSession; open: boolean; close: () => void }) {
  const chat = useLiteAdmin(session);
  const [input, setInput] = useState("");
  const [model, setModel] = useState<string | null>(null);
  const reviewRef = useRef<HTMLDivElement>(null);
  const modelQuery = {
    queryKey: ["liteadmin-models", session.accessToken, session.managementBaseUrl],
    queryFn: () => fetchAvailableModels(session.accessToken),
    enabled: open,
    select: (available: ModelGroup[]) =>
      available.filter((item) => isModeCompatibleWithEndpoint(item.mode, EndpointType.CHAT)),
  };
  const models = useQuery(modelQuery);
  const selectedModel = models.data?.some((item) => item.model_group === model) ? model : null;
  const busy = chat.phase !== "idle";
  const tooLong = input.trim().length > MAX_INPUT_LENGTH;
  const hasValidInput = selectedModel && input.trim() && !tooLong;
  const submitDisabled = busy || models.isError || !hasValidInput;
  const send = () => {
    if (submitDisabled || !selectedModel) return;
    void chat.send(input, selectedModel);
    setInput("");
  };
  return (
    <PopoverContent
      side="top"
      align="end"
      sideOffset={12}
      className={PANEL_CLASS}
      initialFocus={chat.phase === "review" ? reviewRef : true}
    >
      <PanelHeader close={close}>
        <Button
          variant="ghost"
          size="icon-sm"
          aria-label="New chat"
          title="New chat"
          disabled={chat.phase === "applying"}
          onClick={() => {
            chat.reset();
            setInput("");
          }}
        >
          <RotateCcw className="size-4" />
        </Button>
      </PanelHeader>
      <LiteAdminConversation
        entries={chat.entries}
        thinking={chat.phase === "thinking"}
        open={open}
        reviewRef={reviewRef}
        onAnswer={chat.answer}
      />
      <div className="space-y-3 border-t p-3">
        {models.isPending ? (
          <Skeleton className="h-8" aria-label="Loading models" />
        ) : (
          <SearchSelect
            aria-label="LiteAdmin model"
            options={(models.data ?? []).map((item) => ({ value: item.model_group, label: item.model_group }))}
            value={selectedModel}
            onValueChange={setModel}
            placeholder="Choose a chat model"
            disabled={busy}
            allowClear={false}
          />
        )}
        {models.isError && (
          <Alert variant="destructive">
            <AlertDescription>
              Could not load models.{" "}
              <Button variant="link" onClick={() => void models.refetch()}>
                Retry
              </Button>
            </AlertDescription>
          </Alert>
        )}
        {models.isSuccess && models.data.length === 0 && (
          <Alert role="status">
            <AlertDescription>Add a chat model to your gateway to use LiteAdmin.</AlertDescription>
          </Alert>
        )}
        {tooLong && <FieldError>Keep your message within {MAX_INPUT_LENGTH.toLocaleString()} characters.</FieldError>}
        <ChatComposer
          value={input}
          onChange={setInput}
          onSubmit={send}
          onCancel={chat.phase === "applying" ? undefined : chat.stop}
          placeholder="Ask LiteAdmin…"
          disabled={busy || !selectedModel}
          isLoading={busy}
          submitDisabled={submitDisabled}
        />
        <p className="text-xs text-muted-foreground">Use a model you trust with your gateway data.</p>
      </div>
    </PopoverContent>
  );
}

function PanelHeader({ close, children }: { close: () => void; children?: ReactNode }) {
  return (
    <div className="flex items-start justify-between gap-3 border-b p-4">
      <PopoverHeader>
        <PopoverTitle>LiteAdmin</PopoverTitle>
        <PopoverDescription>Ask about your gateway. Review changes in chat.</PopoverDescription>
      </PopoverHeader>
      <div className="flex shrink-0">
        {children}
        <Button variant="ghost" size="icon-sm" aria-label="Close LiteAdmin" onClick={close}>
          <X className="size-4" />
        </Button>
      </div>
    </div>
  );
}
