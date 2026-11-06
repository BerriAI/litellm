"use client";

import { useState, type ReactNode } from "react";
import {
  ArrowDownToLine,
  ArrowUpFromLine,
  Brain,
  Clock,
  Coins,
  DatabaseZap,
  Gauge,
  RotateCw,
  Server,
  Shuffle,
  Timer,
} from "lucide-react";
import type { TFunction } from "i18next";
import { useTranslation } from "react-i18next";
import i18n from "@/i18n/i18n";
import type { MessageType } from "@/components/chat_ui/types";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import { Badge } from "@/components/ui/badge";
import { Empty, EmptyDescription, EmptyHeader, EmptyTitle } from "@/components/ui/empty";
import { FieldDescription } from "@/components/ui/field";
import { Item, ItemActions, ItemContent, ItemDescription, ItemGroup, ItemTitle } from "@/components/ui/item";
import { ScrollArea, ScrollBar } from "@/components/ui/scroll-area";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { formatCost, formatSeconds, servedModel } from "./Conversation";
import { CopyIconButton, FallbackBadge, LongValue } from "./ModelName";

export type CodeSdk = "openai" | "azure";

function TraceRow({ title, description, children }: { title: string; description?: string; children?: ReactNode }) {
  return (
    <Item size="xs" className="flex-nowrap px-0 py-1.5">
      <ItemContent className="flex-none gap-0">
        <ItemTitle className="font-normal whitespace-nowrap text-muted-foreground">{title}</ItemTitle>
        {description && <ItemDescription className="whitespace-nowrap">{description}</ItemDescription>}
      </ItemContent>
      {children && <ItemActions className="min-w-0 flex-1 justify-end text-sm tabular-nums">{children}</ItemActions>}
    </Item>
  );
}

type StatValue = Readonly<{ label: string; icon: ReactNode; value: string | undefined }>;

function StatGrid({ stats }: { stats: readonly StatValue[] }) {
  const shown = stats.filter((stat): stat is StatValue & { value: string } => stat.value !== undefined);
  if (shown.length === 0) return null;
  return (
    <ItemGroup className="grid grid-cols-3 gap-x-4 gap-y-3 py-2 has-data-[size=sm]:gap-y-3">
      {shown.map((stat) => (
        <Item key={stat.label} size="sm" className="flex-col flex-nowrap items-start gap-0.5 p-0">
          <ItemDescription className="inline-flex items-center gap-1 whitespace-nowrap [&_svg]:size-3.5">
            {stat.icon}
            {stat.label}
          </ItemDescription>
          <ItemTitle className="tabular-nums">{stat.value}</ItemTitle>
        </Item>
      ))}
    </ItemGroup>
  );
}

function TraceSection({ value, title, children }: { value: string; title: string; children: ReactNode }) {
  return (
    <AccordionItem value={value} className="px-4">
      <AccordionTrigger className="py-3">{title}</AccordionTrigger>
      <AccordionContent className="pb-3 [&_p:not(:last-child)]:mb-0">
        <ItemGroup className="gap-0 pb-1 has-data-[size=xs]:gap-0">{children}</ItemGroup>
      </AccordionContent>
    </AccordionItem>
  );
}

function Muted({ children }: { children: ReactNode }) {
  return <p className="py-1.5 text-sm text-muted-foreground">{children}</p>;
}

const count = (value: number | undefined): string | undefined => (value === undefined ? undefined : String(value));
const millis = (value: number | undefined): string | undefined =>
  value === undefined ? undefined : `${Math.round(value)}ms`;

function Routing({ message }: { message: MessageType }) {
  const { t } = useTranslation();
  const trace = message.trace;
  const served = servedModel(message);
  const fellBack = (trace?.attemptedFallbacks ?? 0) > 0;
  const reported = Boolean(trace?.callId || trace?.modelGroup || trace?.providerModel);
  return (
    <TraceSection value="routing" title={t("playground.chat.inspector.routing.title")}>
      {trace?.requestedModel && (
        <TraceRow title={t("playground.chat.inspector.routing.requested")}>
          <LongValue
            value={trace.requestedModel}
            detail={t("playground.chat.inspector.routing.requestedDetail")}
            copyLabel={t("playground.chat.inspector.routing.modelNameCopy")}
            className="text-xs"
          />
        </TraceRow>
      )}
      {served && (
        <TraceRow title={t("playground.chat.inspector.routing.servedBy")}>
          {fellBack && <FallbackBadge />}
          <LongValue
            value={served}
            detail={t("playground.chat.inspector.routing.servedByDetail")}
            copyLabel={t("playground.chat.inspector.routing.modelNameCopy")}
            className="text-xs"
          />
        </TraceRow>
      )}
      {trace?.providerModel && (
        <TraceRow title={t("playground.chat.inspector.routing.providerModel")}>
          <LongValue
            value={trace.providerModel}
            detail={
              trace.deploymentId
                ? t("playground.chat.inspector.routing.deploymentDetail", { deploymentId: trace.deploymentId })
                : t("playground.chat.inspector.routing.providerModelDetail")
            }
            copyLabel={t("playground.chat.inspector.routing.providerModelCopy")}
            className="text-xs"
            align="end"
          />
        </TraceRow>
      )}
      {trace?.apiBase && (
        <TraceRow title={t("playground.chat.inspector.routing.apiBase")}>
          <LongValue
            value={trace.apiBase}
            detail={t("playground.chat.inspector.routing.apiBaseDetail")}
            copyLabel={t("playground.chat.inspector.routing.apiBaseCopy")}
            className="text-xs"
            align="end"
          />
        </TraceRow>
      )}
      {trace?.callId && (
        <TraceRow title={t("playground.chat.inspector.routing.callId")}>
          <span className="min-w-0 truncate font-mono text-xs">{trace.callId}</span>
          <CopyIconButton value={trace.callId} label={t("playground.chat.inspector.routing.callIdCopy")} />
        </TraceRow>
      )}
      {trace?.responseId && (
        <TraceRow title={t("playground.chat.inspector.routing.responseId")}>
          <span className="min-w-0 truncate font-mono text-xs">{trace.responseId}</span>
          <CopyIconButton value={trace.responseId} label={t("playground.chat.inspector.routing.responseIdCopy")} />
        </TraceRow>
      )}
      <StatGrid
        stats={[
          {
            label: t("playground.chat.inspector.routing.fallbacks"),
            icon: <Shuffle />,
            value: count(trace?.attemptedFallbacks),
          },
          {
            label: t("playground.chat.inspector.routing.retries"),
            icon: <RotateCw />,
            value: count(trace?.attemptedRetries),
          },
          { label: t("playground.chat.inspector.routing.overhead"), icon: <Gauge />, value: millis(trace?.overheadMs) },
        ]}
      />
      {!reported && <Muted>{t("playground.chat.inspector.routing.notReported")}</Muted>}
    </TraceSection>
  );
}

const noneRan = (policies: readonly string[] | undefined, t: TFunction = i18n.t.bind(i18n)): string | null => {
  if (policies === undefined) return t("playground.chat.inspector.guardrails.noGuardrailsRan");
  return policies.length === 0 ? t("playground.chat.inspector.guardrails.noneRan") : null;
};

function Guardrails({ message }: { message: MessageType }) {
  const { t } = useTranslation();
  const guardrails = message.trace?.appliedGuardrails;
  const policies = message.trace?.appliedPolicies;
  if (guardrails === undefined) {
    return (
      <TraceSection value="guardrails" title={t("playground.chat.inspector.guardrails.title")}>
        <Muted>{t("playground.chat.inspector.guardrails.unsupported")}</Muted>
      </TraceSection>
    );
  }
  const emptyNote = guardrails.length === 0 ? noneRan(policies, t) : null;
  return (
    <TraceSection value="guardrails" title={t("playground.chat.inspector.guardrails.title")}>
      {guardrails.map((name) => (
        <TraceRow key={`g-${name}`} title={name} description={t("playground.chat.inspector.guardrails.guardrail")}>
          <Badge variant="secondary">{t("playground.chat.inspector.guardrails.ran")}</Badge>
        </TraceRow>
      ))}
      {policies?.map((name) => (
        <TraceRow key={`p-${name}`} title={name} description={t("playground.chat.inspector.guardrails.policy")}>
          <Badge variant="secondary">{t("playground.chat.inspector.guardrails.ran")}</Badge>
        </TraceRow>
      ))}
      {emptyNote && <Muted>{emptyNote}</Muted>}
      {policies === undefined && <Muted>{t("playground.chat.inspector.guardrails.policiesUnreported")}</Muted>}
    </TraceSection>
  );
}

function SpeedAndCost({ message }: { message: MessageType }) {
  const { t } = useTranslation();
  const usage = message.usage;
  const cost = usage?.cost ?? message.trace?.cost;
  return (
    <TraceSection value="performance" title={t("playground.chat.inspector.performance.title")}>
      <StatGrid
        stats={[
          {
            label: t("playground.chat.inspector.performance.firstToken"),
            icon: <Timer />,
            value: message.timeToFirstToken === undefined ? undefined : formatSeconds(message.timeToFirstToken),
          },
          {
            label: t("playground.chat.inspector.performance.total"),
            icon: <Clock />,
            value: message.totalLatency === undefined ? undefined : formatSeconds(message.totalLatency),
          },
          {
            label: t("playground.chat.inspector.performance.cost"),
            icon: <Coins />,
            value: cost === undefined ? undefined : formatCost(cost),
          },
          {
            label: t("playground.chat.inspector.performance.tokensIn"),
            icon: <ArrowDownToLine />,
            value: count(usage?.promptTokens),
          },
          {
            label: t("playground.chat.inspector.performance.tokensOut"),
            icon: <ArrowUpFromLine />,
            value: count(usage?.completionTokens),
          },
          {
            label: t("playground.chat.inspector.performance.cached"),
            icon: <DatabaseZap />,
            value: count(usage?.cacheReadTokens),
          },
          {
            label: t("playground.chat.inspector.performance.reasoning"),
            icon: <Brain />,
            value: count(usage?.reasoningTokens),
          },
          {
            label: t("playground.chat.inspector.performance.gatewayTime"),
            icon: <Server />,
            value: millis(message.trace?.responseMs),
          },
        ]}
      />
    </TraceSection>
  );
}

function NothingYet() {
  const { t } = useTranslation();
  return (
    <Empty className="flex-1">
      <EmptyHeader>
        <EmptyTitle className="text-sm">{t("playground.chat.inspector.empty.title")}</EmptyTitle>
        <EmptyDescription>{t("playground.chat.inspector.empty.description")}</EmptyDescription>
      </EmptyHeader>
    </Empty>
  );
}

function Trace({ message }: { message: MessageType | null }) {
  if (!message) return <NothingYet />;
  return (
    <Accordion multiple defaultValue={["routing", "guardrails", "performance"]}>
      <Routing message={message} />
      <Guardrails message={message} />
      <SpeedAndCost message={message} />
    </Accordion>
  );
}

function CodeTab({ code }: { code: (sdk: CodeSdk) => string }) {
  const { t } = useTranslation();
  const [sdk, setSdk] = useState<CodeSdk>("openai");
  const snippet = code(sdk);
  return (
    <div className="flex flex-col">
      <div className="flex items-center gap-2 px-4 pt-3">
        <ToggleGroup
          value={[sdk]}
          onValueChange={(value) => value[0] && setSdk(value[0] as CodeSdk)}
          variant="outline"
          size="sm"
          spacing={0}
          aria-label={t("playground.chat.inspector.code.sdk")}
        >
          <ToggleGroupItem value="openai" className="h-7 px-2 text-xs">
            {t("playground.chat.inspector.code.openai")}
          </ToggleGroupItem>
          <ToggleGroupItem value="azure" className="h-7 px-2 text-xs">
            {t("playground.chat.inspector.code.azure")}
          </ToggleGroupItem>
        </ToggleGroup>
        <span className="ml-auto">
          <CopyIconButton value={snippet} label={t("playground.chat.inspector.code.copyLabel")} />
        </span>
      </div>
      <FieldDescription className="px-4 pt-2 text-xs">
        {t("playground.chat.inspector.code.description")}
      </FieldDescription>
      <ScrollArea className="w-full">
        <pre className="px-4 py-3 font-mono text-xs leading-5 whitespace-pre text-foreground">{snippet}</pre>
        <ScrollBar orientation="horizontal" />
      </ScrollArea>
    </div>
  );
}

export function InspectorPanel({ message, code }: { message: MessageType | null; code: (sdk: CodeSdk) => string }) {
  const { t } = useTranslation();
  return (
    <Tabs defaultValue="trace" className="min-h-0 flex-1 gap-0">
      <div className="flex h-12 shrink-0 items-center border-b border-border px-4">
        <TabsList className="w-full">
          <TabsTrigger value="trace">{t("playground.chat.inspector.tabs.trace")}</TabsTrigger>
          <TabsTrigger value="code">{t("playground.chat.inspector.tabs.code")}</TabsTrigger>
        </TabsList>
      </div>
      <ScrollArea className="min-h-0 flex-1">
        <TabsContent value="trace" className="flex min-h-96 flex-col">
          <Trace message={message} />
        </TabsContent>
        <TabsContent value="code">
          <CodeTab code={code} />
        </TabsContent>
      </ScrollArea>
    </Tabs>
  );
}
