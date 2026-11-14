"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
  type ComponentProps,
  type ReactNode,
  type RefObject,
} from "react";
import { createPortal } from "react-dom";
import { useQuery, type UseQueryResult } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { uiHref } from "@/utils/uiHref";
import { sidebarMenuButtonVariants } from "@/components/shared/Sidebar";
import { RotateCcw } from "lucide-react";
import { z } from "zod";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { useDisableLiteAdmin } from "@/app/(dashboard)/hooks/useDisableLiteAdmin";
import { useUISettings } from "@/app/(dashboard)/hooks/uiSettings/useUISettings";
import { useProxySettingsQuery } from "@/app/(dashboard)/hooks/proxySettings/useProxySettings";
import { ChatComposer } from "@/app/(dashboard)/playground/components/chat_ui/ChatComposer";
import { EndpointType, isModeCompatibleWithEndpoint } from "@/components/chat_ui/mode_endpoint_mapping";
import { fetchAvailableModels, type ModelGroup } from "@/components/llm_calls/fetch_models";
import { createGatewayClient } from "@/components/llm_calls/gateway_client";
import { getProxyBaseUrl } from "@/components/networking";
import { SearchSelect } from "@/components/shared/SearchSelect";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Button, buttonVariants } from "@/components/ui/button";
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { FieldError } from "@/components/ui/field";
import { Kbd } from "@/components/ui/kbd";
import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/cva.config";
import { hasProxyWideSpendView, isProxyAdminRole } from "@/utils/roles";
import { getPreferredLiteAdminModel, MAX_INPUT_LENGTH, resolveInferenceTarget } from "./agent";
import { AriaAvatar } from "./AriaAvatar";
import { AriaStart } from "./AriaStart";
import { ARIA_AGENT_LABEL } from "./aria-brand";
import { LiteAdminConversation } from "./LiteAdminConversation";
import { useLiteAdmin, type ConversationEntry, type LiteAdminSession } from "./useLiteAdmin";

type ManagementSession = Omit<LiteAdminSession, "inferenceBaseUrl">;
type LiteAdminState = {
  access: LiteAdminSession["access"];
  pageContainer: HTMLDivElement | null;
  setPageContainer: (container: HTMLDivElement | null) => void;
};

const LiteAdminContext = createContext<LiteAdminState | null>(null);
function useLiteAdminSession() {
  const auth = useAuthorized();
  const [disabled] = useDisableLiteAdmin(auth.userId);
  const sessionReady = !auth.isLoading && auth.isAuthorized;
  const { data: uiSettings } = useUISettings({ staleTime: 0, refetchInterval: 30_000, enabled: sessionReady });
  const admin = auth.premiumUser === true && !auth.isViewOnly && isProxyAdminRole(auth.userRole);
  if (!uiSettings || uiSettings.values?.disable_aria === true) return null;
  if (!sessionReady || (admin && disabled)) return null;
  if (!auth.token || !auth.accessToken || !auth.userId) return null;
  const access: LiteAdminSession["access"] = admin
    ? { mode: "admin" }
    : {
        mode: "usage",
        scope: hasProxyWideSpendView(auth.userRole) ? { type: "global" } : { type: "user", userId: auth.userId },
      };
  const session = {
    userId: auth.userId,
    token: auth.token,
    accessToken: auth.accessToken,
    managementBaseUrl: getProxyBaseUrl(),
    access,
  };
  return {
    session,
    key: JSON.stringify([auth.userId, session.token, session.accessToken, session.managementBaseUrl, access]),
  };
}

export function LiteAdminFrame({ children }: { children: ReactNode }) {
  const configured = useLiteAdminSession();
  const sessionKey = configured?.key ?? null;
  const router = useRouter();
  const [started, setStarted] = useState(false);
  const [pageContainer, updatePageContainer] = useState<HTMLDivElement | null>(null);
  const setPageContainer = useCallback((container: HTMLDivElement | null) => {
    updatePageContainer(container);
    if (container) setStarted(true);
  }, []);
  const [openedSession, setOpenedSession] = useState(sessionKey);
  if (openedSession !== sessionKey) {
    setOpenedSession(sessionKey);
    setStarted(Boolean(pageContainer));
  }
  useEffect(() => {
    if (sessionKey === null) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key.toLowerCase() !== "j" || !(event.metaKey || event.ctrlKey) || event.shiftKey || event.altKey)
        return;
      event.preventDefault();
      if (pageContainer) {
        pageContainer.querySelector<HTMLTextAreaElement>("textarea:enabled")?.focus();
      } else {
        router.push(uiHref("aria"));
      }
    };
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, [sessionKey, pageContainer, router]);
  return (
    <LiteAdminContext.Provider
      value={
        configured
          ? {
              access: configured.session.access,
              pageContainer,
              setPageContainer,
            }
          : null
      }
    >
      {children}
      {configured && started && (
        <AriaSession key={configured.key} session={configured.session} open={Boolean(pageContainer)} />
      )}
    </LiteAdminContext.Provider>
  );
}

export function AriaSidebarLink({ active }: { active: boolean }) {
  const state = useContext(LiteAdminContext);
  if (!state) return null;
  return (
    <Link
      href={uiHref("aria")}
      aria-current={active ? "page" : undefined}
      aria-label={ARIA_AGENT_LABEL}
      title={ARIA_AGENT_LABEL}
      className={cn(
        sidebarMenuButtonVariants({ isActive: active }),
        "group-data-[collapsed=true]/sidebar:justify-center",
      )}
    >
      <AriaAvatar size={18} />
      <span className="flex-1 group-data-[collapsed=true]/sidebar:hidden">{ARIA_AGENT_LABEL}</span>
      <Kbd className="bg-transparent text-[10px] group-data-[collapsed=true]/sidebar:hidden">⌘J</Kbd>
    </Link>
  );
}

export function LiteAdminPage() {
  const { t } = useTranslation();
  const setPageContainer = useContext(LiteAdminContext)?.setPageContainer;
  if (!setPageContainer) return <p className="p-6 text-sm text-muted-foreground">{t("aria.page.unavailable")}</p>;
  return <div ref={setPageContainer} className="flex min-h-0 flex-1 flex-col" />;
}

export function AriaUsageTrigger() {
  const { t } = useTranslation();
  const state = useContext(LiteAdminContext);
  if (!state) return null;
  return (
    <Link href={uiHref("aria")} className={cn(buttonVariants({ variant: "outline", size: "sm" }), "gap-1.5")}>
      <AriaAvatar size={18} />
      {t("aria.usageTrigger.ask")}
    </Link>
  );
}

function AriaSession({ session, open }: { session: ManagementSession; open: boolean }) {
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
    />
  );
}

function Destination({
  session,
  target,
  loading,
  retry,
  open,
}: {
  session: ManagementSession;
  target: ReturnType<typeof resolveInferenceTarget> | null;
  loading: boolean;
  retry: () => void;
  open: boolean;
}) {
  const { t } = useTranslation();
  const [approved, setApproved] = useState(false);
  if (target?.baseUrl && (!target.requiresConsent || approved)) {
    return <LiteAdminChat session={{ ...session, inferenceBaseUrl: target.baseUrl }} open={open} />;
  }
  return (
    <Panel
      composer={
        <div className="p-4">
          {loading && <Skeleton className="h-24" aria-label={t("aria.destination.loadingSettings")} />}
          {!loading && !target?.baseUrl && (
            <Alert variant="destructive">
              <AlertDescription>{target?.error || t("aria.destination.settingsError")}</AlertDescription>
              <Button variant="link" onClick={retry}>
                {t("aria.destination.retry")}
              </Button>
            </Alert>
          )}
          {!loading && target?.baseUrl && (
            <Card size="sm">
              <CardHeader>
                <CardTitle>{t("aria.destination.consentTitle")}</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <p className="break-all font-mono text-xs">{target.baseUrl}</p>
                <p className="text-muted-foreground">{t("aria.destination.consentBody")}</p>
              </CardContent>
              <CardFooter>
                <Button onClick={() => setApproved(true)}>{t("aria.destination.useConfigured")}</Button>
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

function Panel({
  composer,
  welcome,
  empty = false,
  children,
}: {
  composer: ReactNode;
  welcome?: ReactNode;
  empty?: boolean;
  children: ReactNode;
}) {
  const container = useContext(LiteAdminContext)?.pageContainer;
  if (!container) return null;
  return createPortal(
    <section
      aria-label={ARIA_AGENT_LABEL}
      className="mx-auto flex h-full min-h-0 w-full max-w-3xl flex-col px-2 sm:px-4"
    >
      {empty ? (
        welcome
      ) : (
        <>
          {children}
          {composer}
        </>
      )}
    </section>,
    container,
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
    if (!foreground) {
      lastReviewId.current = null;
      return;
    }
    if (lastReviewId.current === pendingReviewId) return;
    lastReviewId.current = pendingReviewId;
    if (foreground && pendingReviewId) reviewRef.current?.focus({ preventScroll: true });
  }, [foreground, pendingReviewId, reviewRef]);
}

function LiteAdminChat({ session, open }: { session: LiteAdminSession; open: boolean }) {
  const { t } = useTranslation();
  const state = useContext(LiteAdminContext);
  const chat = useLiteAdmin(session);
  const [input, setInput] = useState("");
  const [model, setModel] = useState<string | null>(null);
  const reviewRef = useRef<HTMLDivElement>(null);
  useForegroundReviewFocus(chat.entries, open, reviewRef);
  const modelQuery = {
    queryKey: ["liteadmin-models", session.accessToken, session.managementBaseUrl, session.inferenceBaseUrl],
    queryFn: async ({ signal }: { signal: AbortSignal }) => {
      const gatewayOptions = {
        accessToken: session.accessToken,
        baseURL: session.inferenceBaseUrl,
        maxRetries: 0,
        timeout: 10_000,
      };
      const [catalog, callable] = await Promise.all([
        fetchAvailableModels(session.accessToken),
        createGatewayClient(gatewayOptions).models.list({ signal, fetchOptions: { redirect: "error" } }),
      ]);
      const metadata = new Map(catalog.map((item) => [item.model_group, item]));
      return z
        .array(z.object({ id: z.string(), mode: z.string().nullish() }))
        .parse(callable.data)
        .map(({ id, mode }) => ({
          ...metadata.get(id),
          model_group: id,
          mode: mode ?? metadata.get(id)?.mode,
        }))
        .sort((a, b) => a.model_group.localeCompare(b.model_group));
    },
    enabled: open,
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
  const questionsDisabled = busy || models.isError || !selectedModel;
  const send = (message: string) => {
    const invalidMessage = !message.trim() || message.trim().length > MAX_INPUT_LENGTH;
    const canSend = !busy && !models.isError && !invalidMessage;
    if (!canSend || !selectedModel) return;
    void chat.send(message, selectedModel);
    setInput("");
  };
  const selectQuestion = (question: string) => {
    setInput(question);
    state?.pageContainer?.querySelector<HTMLTextAreaElement>("textarea:enabled")?.focus();
  };
  const startOver = () => {
    if (chat.phase === "applying") return;
    chat.reset();
    setInput("");
  };
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
      tooLong={tooLong}
    />
  );
  return (
    <Panel
      composer={composer}
      empty={chat.entries.length === 0}
      welcome={
        <AriaStart
          composer={composer}
          onQuestion={selectQuestion}
          disabled={questionsDisabled}
          access={session.access}
        />
      }
    >
      <PanelHeader>
        {(chat.entries.length > 0 || input.length > 0) && (
          <Button
            variant="ghost"
            size="sm"
            disabled={chat.phase === "applying"}
            onClick={startOver}
            className="text-muted-foreground"
          >
            <RotateCcw className="size-3.5" />
            {t("aria.chat.startOver")}
          </Button>
        )}
      </PanelHeader>
      <LiteAdminConversation
        entries={chat.entries}
        thinking={chat.phase === "thinking"}
        open={open}
        reviewRef={reviewRef}
        onAnswer={chat.answer}
      />
    </Panel>
  );
}

function LiteAdminComposer({
  models,
  model,
  onModelChange,
  tooLong,
  ...props
}: Omit<ComponentProps<typeof ChatComposer>, "tools" | "placeholder"> & {
  models: UseQueryResult<ModelGroup[]>;
  model: string | null;
  onModelChange: (value: string | null) => void;
  tooLong: boolean;
}) {
  const { t } = useTranslation();
  return (
    <div className="space-y-3 px-4 pt-2 pb-5 sm:px-6">
      <ModelStatus models={models} />
      {tooLong && <FieldError>{t("aria.composer.tooLong", { limit: MAX_INPUT_LENGTH.toLocaleString() })}</FieldError>}
      <ChatComposer
        {...props}
        placeholder={t("aria.composer.placeholder")}
        className="[&>div>[data-slot=input-group]]:min-h-0 [&>div>[data-slot=input-group]]:rounded-2xl [&>div>[data-slot=input-group]]:border-foreground/15 [&>div>[data-slot=input-group]]:shadow-none [&>div>[data-slot=input-group]]:ring-0 dark:[&>div>[data-slot=input-group]]:bg-foreground/[0.03] [&_textarea]:min-h-20 [&_textarea]:px-5 [&_textarea]:pt-5 [&_textarea]:text-base [&_textarea]:placeholder:text-muted-foreground"
        tools={
          <ModelPicker
            models={models}
            value={model}
            onChange={onModelChange}
            disabled={Boolean(props.isLoading)}
            compact
          />
        }
      />
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
  const { t } = useTranslation();
  if (models.isPending) return <Skeleton className="h-8 w-full" aria-label={t("aria.modelPicker.loading")} />;
  return (
    <SearchSelect
      aria-label={t("aria.modelPicker.ariaLabel", { agent: ARIA_AGENT_LABEL })}
      options={(models.data ?? []).map((item) => ({ value: item.model_group, label: item.model_group }))}
      value={value}
      onValueChange={onChange}
      placeholder={t("aria.modelPicker.placeholder")}
      emptyText={models.data?.length ? t("aria.modelPicker.noMatch") : t("aria.modelPicker.noModels")}
      disabled={disabled}
      allowClear={false}
      className={compact ? "border-transparent shadow-none dark:bg-transparent" : undefined}
    />
  );
}

function ModelStatus({ models }: { models: UseQueryResult<ModelGroup[]> }) {
  const { t } = useTranslation();
  if (!models.isError) return null;
  return (
    <Alert variant="destructive">
      <AlertDescription>
        {t("aria.modelStatus.loadError")}{" "}
        <Button variant="link" onClick={() => void models.refetch()}>
          {t("aria.modelStatus.retry")}
        </Button>
      </AlertDescription>
    </Alert>
  );
}

function PanelHeader({ children }: { children?: ReactNode }) {
  return <div className="flex h-12 shrink-0 items-center justify-end px-4 sm:px-6">{children}</div>;
}
