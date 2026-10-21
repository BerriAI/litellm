"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
  type ComponentProps,
  type ReactNode,
  type RefObject,
} from "react";
import { createPortal } from "react-dom";
import { useQuery, type UseQueryResult } from "@tanstack/react-query";
import { History, Plus, Sparkles, X } from "lucide-react";
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
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { cn } from "@/lib/cva.config";
import { isProxyAdminRole } from "@/utils/roles";
import { getPreferredLiteAdminModel, MAX_INPUT_LENGTH, resolveInferenceTarget } from "./agent";
import { LiteAdminConversation } from "./LiteAdminConversation";
import { LiteAdminHomeComposer } from "./LiteAdminHomeComposer";
import { useLiteAdmin, type ConversationEntry, type LiteAdminSession } from "./useLiteAdmin";

type ManagementSession = Omit<LiteAdminSession, "inferenceBaseUrl">;
type ChatPhase = ReturnType<typeof useLiteAdmin>["phase"];
type ChatTab = { id: number; phase: ChatPhase };
type LiteAdminState = {
  open: boolean;
  toggle: () => void;
  show: () => void;
  homeContainer: HTMLDivElement | null;
  setHomeContainer: (container: HTMLDivElement | null) => void;
};

const LiteAdminContext = createContext<LiteAdminState | null>(null);

function useLiteAdminSession() {
  const auth = useAuthorized();
  const [disabled] = useDisableLiteAdmin(auth.userId);
  const sessionReady = !auth.isLoading && auth.isAuthorized && auth.premiumUser === true;
  const writableAdmin = !auth.isViewOnly && isProxyAdminRole(auth.userRole);
  const allowed = sessionReady && writableAdmin && !disabled;
  if (!allowed || !auth.token || !auth.accessToken) return null;
  const session = { token: auth.token, accessToken: auth.accessToken, managementBaseUrl: getProxyBaseUrl() };
  return { session, key: JSON.stringify([auth.userId, session.token, session.accessToken, session.managementBaseUrl]) };
}

/** Wraps the dashboard content column and docks the LiteAdmin panel beside it, so opening it narrows the page instead of covering it. */
export function LiteAdminFrame({ children }: { children: ReactNode }) {
  const configured = useLiteAdminSession();
  const sessionKey = configured?.key ?? null;
  const [open, setOpen] = useState(false);
  const [homeContainer, setHomeContainer] = useState<HTMLDivElement | null>(null);
  const [openedSession, setOpenedSession] = useState(sessionKey);
  if (openedSession !== sessionKey) {
    setOpenedSession(sessionKey);
    setOpen(false);
  }
  const toggle = () => setOpen((current) => !current);
  const show = () => {
    if (!open) setOpen(true);
  };
  useEffect(() => {
    if (sessionKey === null) return;
    const onKeyDown = (event: KeyboardEvent) => {
      const modifier = event.metaKey || event.ctrlKey;
      const extraModifier = event.shiftKey || event.altKey;
      if (event.key.toLowerCase() !== "j" || !modifier || extraModifier) return;
      event.preventDefault();
      toggle();
    };
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, [sessionKey]);
  return (
    <LiteAdminContext.Provider value={configured ? { open, toggle, show, homeContainer, setHomeContainer } : null}>
      {children}
      {configured && (
        <DockedPanel key={configured.key} session={configured.session} open={open} close={() => setOpen(false)} />
      )}
    </LiteAdminContext.Provider>
  );
}

export function LiteAdminHome() {
  const state = useContext(LiteAdminContext);
  if (!state) return null;
  const { open, show, setHomeContainer } = state;
  return (
    <section aria-label="Ask LiteAdmin" hidden={open} className="space-y-3">
      <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-2">
        <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
          <h2 className="flex items-center gap-2 text-sm font-semibold">
            <Sparkles className="size-4 text-info" />
            LiteAdmin
          </h2>
          <p className="border-l pl-3 text-sm text-muted-foreground">Control your entire gateway.</p>
        </div>
        <Button variant="ghost" size="sm" onClick={show} className="text-muted-foreground">
          <History className="size-3.5" />
          Recent chats
        </Button>
      </div>
      <div ref={setHomeContainer} />
    </section>
  );
}

export default function LiteAdminTrigger() {
  const state = useContext(LiteAdminContext);
  if (!state) return null;
  return (
    <Button
      variant="ghost"
      aria-label="LiteAdmin"
      aria-keyshortcuts="Meta+J Control+J"
      aria-expanded={state.open}
      onClick={state.toggle}
      className="rounded-full bg-info/10 text-info hover:bg-info/15 hover:text-info/80 aria-expanded:bg-info/15 aria-expanded:text-info"
    >
      <Sparkles className="size-4" />
      <span className="hidden lg:inline">LiteAdmin</span>
      <kbd className="hidden rounded-full border border-info/30 px-1.5 font-sans text-xs lg:inline">⌘J</kbd>
    </Button>
  );
}

function DockedPanel({ session, open, close }: { session: ManagementSession; open: boolean; close: () => void }) {
  const settings = useProxySettingsQuery(session.accessToken);
  const candidate =
    settings.data?.LITELLM_UI_API_DOC_BASE_URL?.trim() ||
    settings.data?.PROXY_BASE_URL?.trim() ||
    session.managementBaseUrl;
  const target = settings.data
    ? resolveInferenceTarget(candidate, session.managementBaseUrl, window.location.href)
    : null;
  return (
    <Destination
      key={target?.baseUrl ?? "unavailable"}
      session={session}
      target={target}
      loading={settings.isPending}
      retry={() => void settings.refetch()}
      open={open}
      close={close}
    />
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
    return <LiteAdminChats session={{ ...session, inferenceBaseUrl: target.baseUrl }} open={open} close={close} />;
  }
  return (
    <Panel
      open={open}
      composer={
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
      }
    >
      <PanelHeader close={close} />
    </Panel>
  );
}

/** On open, focus `initialFocus` when given, else the composer when it is usable, else the panel itself. */
function Panel({
  open,
  initialFocus,
  composer,
  children,
}: {
  open: boolean;
  initialFocus?: RefObject<HTMLElement | null>;
  composer: ReactNode;
  children: ReactNode;
}) {
  const state = useContext(LiteAdminContext);
  const homeContainer = !open ? state?.homeContainer : null;
  const ref = useRef<HTMLElement>(null);
  useEffect(() => {
    const panel = ref.current;
    if (!open || !panel) return;
    (initialFocus?.current ?? panel.querySelector<HTMLElement>("textarea:enabled") ?? panel).focus({
      preventScroll: true,
    });
  }, [open, initialFocus]);
  return (
    <aside
      ref={ref}
      aria-label="LiteAdmin"
      tabIndex={-1}
      hidden={!open}
      className={cn(
        "w-[min(26rem,40vw)] min-w-80 flex-none flex-col overflow-hidden border-l bg-background outline-none",
        open && "flex",
      )}
    >
      {children}
      {homeContainer ? createPortal(composer, homeContainer) : composer}
    </aside>
  );
}

function LiteAdminChats({ session, open, close }: { session: LiteAdminSession; open: boolean; close: () => void }) {
  const [tabs, setTabs] = useState<{ chats: ChatTab[]; activeId: number; nextId: number }>({
    chats: [{ id: 1, phase: "idle" }],
    activeId: 1,
    nextId: 2,
  });
  const reviewRef = useRef<HTMLDivElement>(null);
  const activeTabRef = useRef<HTMLButtonElement>(null);
  const focusAfterClose = useRef(false);
  useLayoutEffect(() => {
    if (!focusAfterClose.current) return;
    focusAfterClose.current = false;
    activeTabRef.current?.focus();
  });
  const addChat = () => {
    setTabs((current) => ({
      chats: [...current.chats, { id: current.nextId, phase: "idle" }],
      activeId: current.nextId,
      nextId: current.nextId + 1,
    }));
  };
  const closeChat = (id: number) => {
    if (tabs.chats.some((chat) => chat.id === id && chat.phase === "applying")) return;
    focusAfterClose.current = true;
    setTabs((current) => {
      const closing = current.chats.find((chat) => chat.id === id);
      if (!closing || closing.phase === "applying") return current;
      const remaining = current.chats.filter((chat) => chat.id !== id);
      const next = remaining.find((chat) => chat.id > id) ?? remaining.at(-1);
      if (!next) {
        return {
          chats: [{ id: current.nextId, phase: "idle" }],
          activeId: current.nextId,
          nextId: current.nextId + 1,
        };
      }
      return { ...current, chats: remaining, activeId: current.activeId === id ? next.id : current.activeId };
    });
  };
  const updatePhase = useCallback((id: number, phase: ChatPhase) => {
    setTabs((current) => {
      if (!current.chats.some((chat) => chat.id === id && chat.phase !== phase)) return current;
      return { ...current, chats: current.chats.map((chat) => (chat.id === id ? { ...chat, phase } : chat)) };
    });
  }, []);
  return (
    <Tabs
      value={tabs.activeId}
      onValueChange={(id) => {
        if (typeof id === "number") setTabs((current) => ({ ...current, activeId: id }));
      }}
      className="contents"
    >
      <Panel open={open} initialFocus={reviewRef} composer={null}>
        <PanelHeader close={close}>
          <Button variant="ghost" size="icon-sm" aria-label="New chat" title="New chat" onClick={addChat}>
            <Plus className="size-4" />
          </Button>
        </PanelHeader>
        <TabsList
          aria-label="LiteAdmin chats"
          className="h-auto w-full shrink-0 justify-start overflow-x-auto rounded-none border-b bg-transparent p-2"
        >
          {tabs.chats.map((tab) => (
            <div key={tab.id} className="flex shrink-0 items-center">
              <TabsTrigger
                ref={tabs.activeId === tab.id ? activeTabRef : undefined}
                value={tab.id}
                aria-label={`Chat ${tab.id}`}
                title={tab.phase === "review" ? "Approval needed" : undefined}
                className="h-8 flex-none"
              >
                Chat {tab.id}
                {tab.phase !== "idle" && <span aria-hidden className="size-1.5 rounded-full bg-info" />}
              </TabsTrigger>
              <Button
                variant="ghost"
                size="icon-xs"
                aria-label={`Close Chat ${tab.id}`}
                title={tab.phase === "applying" ? "Wait for this change to finish" : `Close Chat ${tab.id}`}
                disabled={tab.phase === "applying"}
                onClick={() => closeChat(tab.id)}
              >
                <X className="size-3" />
              </Button>
            </div>
          ))}
        </TabsList>
        {tabs.chats.map((tab) => (
          <LiteAdminChat
            key={tab.id}
            id={tab.id}
            session={session}
            active={tabs.activeId === tab.id}
            open={open}
            reviewRef={reviewRef}
            onPhaseChange={updatePhase}
          />
        ))}
      </Panel>
    </Tabs>
  );
}

function useForegroundReviewFocus(
  entries: ConversationEntry[],
  foreground: boolean,
  reviewRef: RefObject<HTMLDivElement | null>,
) {
  const lastReviewId = useRef<string | null>(null);
  const pendingReviewId = entries.find((entry) => entry.kind === "action" && entry.status === "review")?.id ?? null;
  useEffect(() => {
    if (lastReviewId.current === pendingReviewId) return;
    lastReviewId.current = pendingReviewId;
    if (foreground && pendingReviewId) reviewRef.current?.focus({ preventScroll: true });
  }, [foreground, pendingReviewId, reviewRef]);
}

function LiteAdminChat({
  id,
  session,
  active,
  open,
  reviewRef,
  onPhaseChange,
}: {
  id: number;
  session: LiteAdminSession;
  active: boolean;
  open: boolean;
  reviewRef: RefObject<HTMLDivElement | null>;
  onPhaseChange: (id: number, phase: ChatPhase) => void;
}) {
  const state = useContext(LiteAdminContext);
  const chat = useLiteAdmin(session);
  const [input, setInput] = useState("");
  const [model, setModel] = useState<string | null>(null);
  useForegroundReviewFocus(chat.entries, active && open, reviewRef);
  useEffect(() => onPhaseChange(id, chat.phase), [id, chat.phase, onPhaseChange]);
  const modelQuery = {
    queryKey: ["liteadmin-models", session.accessToken, session.managementBaseUrl],
    queryFn: () => fetchAvailableModels(session.accessToken),
    enabled: active && (open || Boolean(state?.homeContainer)),
    select: (available: ModelGroup[]) =>
      available.filter((item) => isModeCompatibleWithEndpoint(item.mode, EndpointType.CHAT)),
  };
  const models = useQuery(modelQuery);
  const preferredModel = model ?? getPreferredLiteAdminModel(models.data ?? []);
  const selectedModel = models.data?.some((item) => item.model_group === preferredModel) ? preferredModel : null;
  const busy = chat.phase !== "idle";
  const tooLong = input.trim().length > MAX_INPUT_LENGTH;
  const hasValidInput = selectedModel && input.trim() && !tooLong;
  const submitDisabled = busy || models.isError || !hasValidInput;
  const send = () => {
    if (submitDisabled || !selectedModel) return;
    state?.show();
    void chat.send(input, selectedModel);
    setInput("");
  };
  if (!active) return null;
  const homeContainer = !open ? state?.homeContainer : null;
  const composerProps = {
    value: input,
    onChange: setInput,
    onSubmit: send,
    onCancel: chat.phase === "applying" ? undefined : chat.stop,
    disabled: busy || !selectedModel,
    isLoading: busy,
    submitDisabled,
  };
  const composer = (
    <LiteAdminComposer
      {...composerProps}
      models={models}
      model={selectedModel}
      onModelChange={setModel}
      home={Boolean(homeContainer)}
      tooLong={tooLong}
    />
  );
  return (
    <TabsContent value={id} className="flex min-h-0 flex-1 flex-col">
      <LiteAdminConversation
        entries={chat.entries}
        thinking={chat.phase === "thinking"}
        open={open}
        reviewRef={reviewRef}
        onAnswer={chat.answer}
      />
      {homeContainer ? createPortal(composer, homeContainer) : composer}
    </TabsContent>
  );
}

function LiteAdminComposer({
  models,
  model,
  onModelChange,
  home,
  tooLong,
  ...props
}: Omit<ComponentProps<typeof LiteAdminHomeComposer>, "modelPicker"> & {
  models: UseQueryResult<ModelGroup[]>;
  model: string | null;
  onModelChange: (value: string | null) => void;
  home: boolean;
  tooLong: boolean;
}) {
  return (
    <div className={cn("space-y-3", !home && "border-t p-3")}>
      {!home && <ModelPicker models={models} value={model} onChange={onModelChange} disabled={props.isLoading} />}
      <ModelStatus models={models} />
      {tooLong && <FieldError>Keep your message within {MAX_INPUT_LENGTH.toLocaleString()} characters.</FieldError>}
      {home ? (
        <LiteAdminHomeComposer
          {...props}
          modelPicker={
            <ModelPicker models={models} value={model} onChange={onModelChange} disabled={props.isLoading} compact />
          }
        />
      ) : (
        <>
          <ChatComposer {...props} placeholder="Ask LiteAdmin…" />
          <p className="text-xs text-muted-foreground">Use a model you trust with your gateway data.</p>
        </>
      )}
    </div>
  );
}

function ModelPicker({
  models,
  value,
  onChange,
  disabled,
  compact = false,
}: {
  models: UseQueryResult<ModelGroup[]>;
  value: string | null;
  onChange: (value: string | null) => void;
  disabled: boolean;
  compact?: boolean;
}) {
  if (models.isPending) return <Skeleton className="h-8 w-full" aria-label="Loading models" />;
  return (
    <SearchSelect
      aria-label="LiteAdmin model"
      options={(models.data ?? []).map((item) => ({ value: item.model_group, label: item.model_group }))}
      value={value}
      onValueChange={onChange}
      placeholder="Choose a chat model"
      disabled={disabled}
      allowClear={false}
      className={compact ? "border-transparent shadow-none dark:bg-transparent" : undefined}
    />
  );
}

function ModelStatus({ models }: { models: UseQueryResult<ModelGroup[]> }) {
  return (
    <>
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
    </>
  );
}

function PanelHeader({ close, children }: { close: () => void; children?: ReactNode }) {
  return (
    <div className="flex items-start justify-between gap-3 border-b p-4">
      <div className="flex flex-col gap-1 text-sm">
        <h2 className="flex items-center gap-1.5 font-medium">
          <Sparkles className="size-4 text-info" />
          LiteAdmin
        </h2>
        <p className="text-muted-foreground">Ask about your gateway. Review changes in chat.</p>
      </div>
      <div className="flex shrink-0">
        {children}
        <Button variant="ghost" size="icon-sm" aria-label="Close LiteAdmin" onClick={close}>
          <X className="size-4" />
        </Button>
      </div>
    </div>
  );
}
