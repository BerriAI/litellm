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
import { History, Plus, X } from "lucide-react";
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
import { Dialog, DialogClose, DialogContent, DialogDescription, DialogTitle } from "@/components/ui/dialog";
import { FieldError } from "@/components/ui/field";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { cn } from "@/lib/cva.config";
import { hasProxyWideSpendView, isProxyAdminRole } from "@/utils/roles";
import { getPreferredLiteAdminModel, MAX_INPUT_LENGTH, resolveInferenceTarget } from "./agent";
import { AriaAvatar } from "./AriaAvatar";
import { ARIA_AGENT_LABEL } from "./aria-brand";
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
  access: LiteAdminSession["access"];
  launcherRef: RefObject<HTMLButtonElement | null>;
  homeContainer: HTMLDivElement | null;
  setHomeContainer: (container: HTMLDivElement | null) => void;
  usageContainer: HTMLDivElement | null;
  setUsageContainer: (container: HTMLDivElement | null) => void;
};

const LiteAdminContext = createContext<LiteAdminState | null>(null);
function useLiteAdminSession() {
  const auth = useAuthorized();
  const [disabled] = useDisableLiteAdmin(auth.userId);
  const sessionReady = !auth.isLoading && auth.isAuthorized;
  const admin = auth.premiumUser === true && !auth.isViewOnly && isProxyAdminRole(auth.userRole);
  if (!sessionReady || (admin && disabled)) return null;
  if (!auth.token || !auth.accessToken || !auth.userId) return null;
  const access: LiteAdminSession["access"] = admin
    ? { mode: "admin" }
    : {
        mode: "usage",
        scope: hasProxyWideSpendView(auth.userRole) ? { type: "global" } : { type: "user", userId: auth.userId },
      };
  const session = { token: auth.token, accessToken: auth.accessToken, managementBaseUrl: getProxyBaseUrl(), access };
  return {
    session,
    key: JSON.stringify([auth.userId, session.token, session.accessToken, session.managementBaseUrl, access]),
  };
}

/** Keeps one Aria conversation available from the dashboard and its floating launcher. */
export function LiteAdminFrame({ children }: { children: ReactNode }) {
  const configured = useLiteAdminSession();
  const sessionKey = configured?.key ?? null;
  const [open, setOpen] = useState(false);
  const launcherRef = useRef<HTMLButtonElement>(null);
  const [started, setStarted] = useState(false);
  const [homeContainer, setHomeContainer] = useState<HTMLDivElement | null>(null);
  const [usageContainer, updateUsageContainer] = useState<HTMLDivElement | null>(null);
  const setUsageContainer = useCallback((container: HTMLDivElement | null) => {
    updateUsageContainer(container);
    if (container) setStarted(true);
  }, []);
  const [openedSession, setOpenedSession] = useState(sessionKey);
  if (openedSession !== sessionKey) {
    setOpenedSession(sessionKey);
    setOpen(false);
    setStarted(Boolean(usageContainer));
  }
  const toggle = () => {
    setStarted(true);
    setOpen((current) => !current);
  };
  const show = () => {
    if (open) return;
    setStarted(true);
    setOpen(true);
  };
  const showPanel = configured?.session.access.mode === "admin" || started;
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
    <LiteAdminContext.Provider
      value={
        configured
          ? {
              open,
              toggle,
              show,
              access: configured.session.access,
              launcherRef,
              homeContainer,
              setHomeContainer,
              usageContainer,
              setUsageContainer,
            }
          : null
      }
    >
      {children}
      {configured && !open && <AriaLauncher show={show} buttonRef={launcherRef} />}
      {configured && showPanel && (
        <AriaPopup key={configured.key} session={configured.session} open={open} close={() => setOpen(false)} />
      )}
    </LiteAdminContext.Provider>
  );
}

function AriaLauncher({ show, buttonRef }: { show: () => void; buttonRef: RefObject<HTMLButtonElement | null> }) {
  return (
    <Button
      ref={buttonRef}
      variant="outline"
      aria-label={`Launch ${ARIA_AGENT_LABEL}`}
      aria-keyshortcuts="Meta+J Control+J"
      aria-expanded={false}
      onClick={show}
      className="fixed bottom-4 right-4 z-floating h-11 gap-2 rounded-full border-blue-200 bg-background pl-2 pr-4 text-blue-700 shadow-md hover:bg-blue-50 hover:text-blue-800 dark:border-blue-400/30 dark:text-blue-300 dark:hover:bg-blue-400/10"
    >
      <AriaAvatar />
      <span>{ARIA_AGENT_LABEL}</span>
    </Button>
  );
}

export function LiteAdminHome() {
  const state = useContext(LiteAdminContext);
  if (!state || state.access.mode !== "admin") return null;
  const { open, show, setHomeContainer } = state;
  return (
    <section aria-label={`Ask ${ARIA_AGENT_LABEL}`} hidden={open} className="space-y-3">
      <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-2">
        <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
          <h2 className="flex items-center gap-2 text-sm font-semibold">
            <AriaAvatar size={32} />
            {ARIA_AGENT_LABEL}
          </h2>
          <p className="border-l pl-3 text-sm text-muted-foreground">Your gateway, a little easier.</p>
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
  if (!state || state.access.mode !== "admin") return null;
  return (
    <Button
      variant="ghost"
      aria-label={ARIA_AGENT_LABEL}
      aria-keyshortcuts="Meta+J Control+J"
      aria-expanded={state.open}
      onClick={state.toggle}
      className="gap-2 rounded-full border border-blue-200/70 bg-blue-50/80 pl-1.5 pr-3 text-blue-700 hover:bg-blue-100 hover:text-blue-800 aria-expanded:bg-blue-100 dark:border-blue-400/20 dark:bg-blue-400/10 dark:text-blue-300 dark:hover:bg-blue-400/20"
    >
      <AriaAvatar />
      <span className="md:hidden lg:inline">{ARIA_AGENT_LABEL}</span>
      <kbd className="hidden rounded-full border border-info/30 px-1.5 font-sans text-xs lg:inline">⌘J</kbd>
    </Button>
  );
}

export function AriaUsageTrigger() {
  const state = useContext(LiteAdminContext);
  if (!state) return null;
  return (
    <Button
      variant="outline"
      size="sm"
      aria-label={`Open ${ARIA_AGENT_LABEL}`}
      aria-expanded={state.open}
      onClick={state.show}
      className="gap-1.5 border-blue-200 bg-blue-50/70 pl-1.5 text-blue-700 hover:bg-blue-100 hover:text-blue-800 dark:border-blue-400/20 dark:bg-blue-400/10 dark:text-blue-300 dark:hover:bg-blue-400/20"
    >
      <AriaAvatar size={24} />
      {ARIA_AGENT_LABEL}
    </Button>
  );
}

export function AriaUsageComposer() {
  const state = useContext(LiteAdminContext);
  if (!state) return null;
  const { open, setUsageContainer } = state;
  return (
    <section aria-label="Ask Aria about usage" hidden={open} className="mb-4">
      <div className="mb-2 flex flex-wrap items-center gap-x-3 gap-y-1">
        <h2 className="flex items-center gap-2 text-sm font-medium">
          <AriaAvatar size={24} /> {ARIA_AGENT_LABEL}
        </h2>
        <p className="text-sm text-muted-foreground">Your AI usage assistant</p>
      </div>
      <div ref={setUsageContainer} />
    </section>
  );
}

function AriaPopup({ session, open, close }: { session: ManagementSession; open: boolean; close: () => void }) {
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
      close={close}
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
                  This gateway uses a different address. Aria will send your existing session credential to the address
                  shown above.
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
      <PanelHeader />
    </Panel>
  );
}

/** On open, focus `initialFocus` when given, else the composer when it is usable, else the panel itself. */
function Panel({
  open,
  close,
  initialFocus,
  composer,
  children,
}: {
  open: boolean;
  close: () => void;
  initialFocus?: RefObject<HTMLElement | null>;
  composer: ReactNode;
  children: ReactNode;
}) {
  const state = useContext(LiteAdminContext);
  const homeContainer = !open ? state?.usageContainer ?? state?.homeContainer : null;
  const ref = useRef<HTMLDivElement>(null);
  return (
    <Dialog
      open={open}
      modal={false}
      disablePointerDismissal
      onOpenChange={(nextOpen) => {
        if (!nextOpen) close();
      }}
    >
      <DialogContent
        ref={ref}
        keepMounted
        showCloseButton={false}
        overlayClassName="hidden"
        hidden={!open}
        aria-label={ARIA_AGENT_LABEL}
        initialFocus={() =>
          initialFocus?.current ?? ref.current?.querySelector<HTMLElement>("textarea:enabled") ?? ref.current
        }
        finalFocus={state?.launcherRef}
        className="bottom-4 right-4 left-auto top-auto flex h-[min(42rem,calc(100dvh-2rem))] translate-x-0 translate-y-0 flex-col gap-0 overflow-hidden p-0 shadow-xl sm:max-w-md"
      >
        {children}
        {homeContainer ? createPortal(composer, homeContainer) : composer}
      </DialogContent>
    </Dialog>
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
      <Panel open={open} close={close} initialFocus={reviewRef} composer={null}>
        <PanelHeader>
          <Button variant="ghost" size="icon-sm" aria-label="New chat" title="New chat" onClick={addChat}>
            <Plus className="size-4" />
          </Button>
        </PanelHeader>
        <TabsList
          aria-label={`${ARIA_AGENT_LABEL} chats`}
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
    enabled: active && (open || Boolean(state?.homeContainer || state?.usageContainer)),
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
  const send = (message: string) => {
    if (busy || models.isError || !selectedModel || !message.trim() || message.trim().length > MAX_INPUT_LENGTH) return;
    state?.show();
    void chat.send(message, selectedModel);
    setInput("");
  };
  if (!active) return null;
  const homeContainer = !open ? state?.usageContainer ?? state?.homeContainer : null;
  const composerProps = {
    value: input,
    onChange: setInput,
    onSubmit: () => send(input),
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
      usage={Boolean(state?.usageContainer)}
      tooLong={tooLong}
    />
  );
  return (
    <TabsContent value={id} className="flex min-h-0 flex-1 flex-col">
      <LiteAdminConversation
        entries={chat.entries}
        thinking={chat.phase === "thinking"}
        readOnly={session.access.mode === "usage"}
        open={open}
        reviewRef={reviewRef}
        onAnswer={chat.answer}
        onQuestion={send}
        questionsDisabled={busy || models.isError || !selectedModel}
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
  usage,
  tooLong,
  ...props
}: Omit<ComponentProps<typeof LiteAdminHomeComposer>, "modelPicker" | "usage"> & {
  models: UseQueryResult<ModelGroup[]>;
  model: string | null;
  onModelChange: (value: string | null) => void;
  home: boolean;
  usage: boolean;
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
          usage={usage}
          modelPicker={
            <ModelPicker models={models} value={model} onChange={onModelChange} disabled={props.isLoading} compact />
          }
        />
      ) : (
        <>
          <ChatComposer {...props} placeholder="Ask Aria…" />
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
      aria-label={`${ARIA_AGENT_LABEL} model`}
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
          <AlertDescription>Add a chat model to your gateway to use Aria.</AlertDescription>
        </Alert>
      )}
    </>
  );
}

function PanelHeader({ children }: { children?: ReactNode }) {
  const state = useContext(LiteAdminContext);
  return (
    <div className="flex items-start justify-between gap-3 border-b p-4">
      <div className="flex flex-col gap-1 text-sm">
        <DialogTitle className="flex items-center gap-2 font-semibold">
          <AriaAvatar size={32} />
          {ARIA_AGENT_LABEL}
        </DialogTitle>
        <DialogDescription>
          {state?.access.mode === "usage"
            ? "Your AI assistant for usage and spend."
            : "Ask about your gateway. Review changes in chat."}
        </DialogDescription>
      </div>
      <div className="flex shrink-0">
        {children}
        <DialogClose render={<Button variant="ghost" size="icon-sm" aria-label={`Close ${ARIA_AGENT_LABEL}`} />}>
          <X className="size-4" />
        </DialogClose>
      </div>
    </div>
  );
}
