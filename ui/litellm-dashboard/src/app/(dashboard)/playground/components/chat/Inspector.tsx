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
  const trace = message.trace;
  const served = servedModel(message);
  const fellBack = (trace?.attemptedFallbacks ?? 0) > 0;
  const reported = Boolean(trace?.callId || trace?.modelGroup || trace?.providerModel);
  return (
    <TraceSection value="routing" title="Routing">
      {trace?.requestedModel && (
        <TraceRow title="Requested">
          <LongValue value={trace.requestedModel} detail="Requested model" copyLabel="model name" className="text-xs" />
        </TraceRow>
      )}
      {served && (
        <TraceRow title="Served by">
          {fellBack && <FallbackBadge />}
          <LongValue value={served} detail="Model group that answered" copyLabel="model name" className="text-xs" />
        </TraceRow>
      )}
      {trace?.providerModel && (
        <TraceRow title="Provider model">
          <LongValue
            value={trace.providerModel}
            detail={trace.deploymentId ? `Deployment ID ${trace.deploymentId}` : "Model sent to the provider"}
            copyLabel="provider model"
            className="text-xs"
            align="end"
          />
        </TraceRow>
      )}
      {trace?.apiBase && (
        <TraceRow title="API base">
          <LongValue
            value={trace.apiBase}
            detail="Provider endpoint"
            copyLabel="API base"
            className="text-xs"
            align="end"
          />
        </TraceRow>
      )}
      {trace?.callId && (
        <TraceRow title="Call ID">
          <span className="min-w-0 truncate font-mono text-xs">{trace.callId}</span>
          <CopyIconButton value={trace.callId} label="call ID" />
        </TraceRow>
      )}
      {trace?.responseId && (
        <TraceRow title="Response ID">
          <span className="min-w-0 truncate font-mono text-xs">{trace.responseId}</span>
          <CopyIconButton value={trace.responseId} label="response ID" />
        </TraceRow>
      )}
      <StatGrid
        stats={[
          { label: "Fallbacks", icon: <Shuffle />, value: count(trace?.attemptedFallbacks) },
          { label: "Retries", icon: <RotateCw />, value: count(trace?.attemptedRetries) },
          { label: "Overhead", icon: <Gauge />, value: millis(trace?.overheadMs) },
        ]}
      />
      {!reported && (
        <Muted>
          The gateway did not share routing details with this page. Open the dashboard from the proxy to see them
        </Muted>
      )}
    </TraceSection>
  );
}

const noneRan = (policies: readonly string[] | undefined): string | null => {
  if (policies === undefined) return "No guardrails ran on this request";
  return policies.length === 0 ? "No guardrails or policies ran on this request" : null;
};

function Guardrails({ message }: { message: MessageType }) {
  const guardrails = message.trace?.appliedGuardrails;
  const policies = message.trace?.appliedPolicies;
  if (guardrails === undefined) {
    return (
      <TraceSection value="guardrails" title="Guardrails and policies">
        <Muted>This endpoint does not report which guardrails or policies ran</Muted>
      </TraceSection>
    );
  }
  const emptyNote = guardrails.length === 0 ? noneRan(policies) : null;
  return (
    <TraceSection value="guardrails" title="Guardrails and policies">
      {guardrails.map((name) => (
        <TraceRow key={`g-${name}`} title={name} description="Guardrail">
          <Badge variant="secondary">Ran</Badge>
        </TraceRow>
      ))}
      {policies?.map((name) => (
        <TraceRow key={`p-${name}`} title={name} description="Policy">
          <Badge variant="secondary">Ran</Badge>
        </TraceRow>
      ))}
      {emptyNote && <Muted>{emptyNote}</Muted>}
      {policies === undefined && <Muted>Policies are only reported when the dashboard is opened from the proxy</Muted>}
    </TraceSection>
  );
}

function SpeedAndCost({ message }: { message: MessageType }) {
  const usage = message.usage;
  const cost = usage?.cost ?? message.trace?.cost;
  return (
    <TraceSection value="performance" title="Speed and cost">
      <StatGrid
        stats={[
          {
            label: "First token",
            icon: <Timer />,
            value: message.timeToFirstToken === undefined ? undefined : formatSeconds(message.timeToFirstToken),
          },
          {
            label: "Total",
            icon: <Clock />,
            value: message.totalLatency === undefined ? undefined : formatSeconds(message.totalLatency),
          },
          { label: "Cost", icon: <Coins />, value: cost === undefined ? undefined : formatCost(cost) },
          { label: "Tokens in", icon: <ArrowDownToLine />, value: count(usage?.promptTokens) },
          { label: "Tokens out", icon: <ArrowUpFromLine />, value: count(usage?.completionTokens) },
          { label: "Cached", icon: <DatabaseZap />, value: count(usage?.cacheReadTokens) },
          { label: "Reasoning", icon: <Brain />, value: count(usage?.reasoningTokens) },
          { label: "Gateway time", icon: <Server />, value: millis(message.trace?.responseMs) },
        ]}
      />
    </TraceSection>
  );
}

function NothingYet() {
  return (
    <Empty className="flex-1">
      <EmptyHeader>
        <EmptyTitle className="text-sm">Nothing to inspect yet</EmptyTitle>
        <EmptyDescription>
          Each reply shows here with the deployment that served it, fallbacks, guardrails, timing and cost
        </EmptyDescription>
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
          aria-label="SDK"
        >
          <ToggleGroupItem value="openai" className="h-7 px-2 text-xs">
            OpenAI SDK
          </ToggleGroupItem>
          <ToggleGroupItem value="azure" className="h-7 px-2 text-xs">
            Azure SDK
          </ToggleGroupItem>
        </ToggleGroup>
        <span className="ml-auto">
          <CopyIconButton value={snippet} label="code" />
        </span>
      </div>
      <FieldDescription className="px-4 pt-2 text-xs">
        Send this request from your own app. Replace YOUR_LITELLM_API_KEY with a virtual key
      </FieldDescription>
      <ScrollArea className="w-full">
        <pre className="px-4 py-3 font-mono text-xs leading-5 whitespace-pre text-foreground">{snippet}</pre>
        <ScrollBar orientation="horizontal" />
      </ScrollArea>
    </div>
  );
}

export function InspectorPanel({ message, code }: { message: MessageType | null; code: (sdk: CodeSdk) => string }) {
  return (
    <Tabs defaultValue="trace" className="min-h-0 flex-1 gap-0">
      <div className="flex h-12 shrink-0 items-center border-b border-border px-4">
        <TabsList className="w-full">
          <TabsTrigger value="trace">Trace</TabsTrigger>
          <TabsTrigger value="code">Code</TabsTrigger>
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
