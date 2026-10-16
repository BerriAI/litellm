"use client";

import { createContext, useContext, useState, type ReactNode } from "react";
import {
  ArrowUp,
  ChevronDown,
  ChevronRight,
  Clock,
  Coins,
  Database,
  DatabaseZap,
  ArrowDownToLine,
  ArrowUpFromLine,
  Minus,
  Plus,
  Timer,
  Copy,
  Paperclip,
  PlayCircle,
  ScanSearch,
  Shield,
  SlidersHorizontal,
  RotateCcw,
  Split,
  SquarePen,
} from "lucide-react";
import { PageTabs, PageTabsList, PageTabsTrigger } from "@/components/shared/Page";
import { PageHeader, PageHeaderDescription, PageHeaderTitle } from "@/components/shared/PageHeader";
import { cn } from "@/lib/cva.config";
import BetaBadge from "@/components/BetaBadge";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import {
  Combobox,
  ComboboxChip,
  ComboboxChips,
  ComboboxChipsInput,
  ComboboxContent,
  ComboboxEmpty,
  ComboboxItem,
  ComboboxList,
  ComboboxValue,
  useComboboxAnchor,
} from "@/components/ui/combobox";
import { ButtonGroup } from "@/components/ui/button-group";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Empty, EmptyDescription, EmptyHeader, EmptyMedia, EmptyTitle } from "@/components/ui/empty";
import { Field, FieldDescription, FieldGroup, FieldLabel, FieldSet } from "@/components/ui/field";
import { HoverCard, HoverCardContent, HoverCardTrigger } from "@/components/ui/hover-card";
import { Input } from "@/components/ui/input";
import { InputGroup, InputGroupAddon, InputGroupButton, InputGroupTextarea } from "@/components/ui/input-group";
import { Item, ItemActions, ItemContent, ItemDescription, ItemGroup, ItemTitle } from "@/components/ui/item";
import { Message, MessageContent, MessageFooter, MessageHeader } from "@/components/ui/message";
import {
  MessageScroller,
  MessageScrollerButton,
  MessageScrollerContent,
  MessageScrollerItem,
  MessageScrollerProvider,
  MessageScrollerViewport,
} from "@/components/ui/message-scroller";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Toggle } from "@/components/ui/toggle";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { ENDPOINT_OPTIONS } from "../chat_ui/chatConstants";
import {
  Markdown,
  StreamingTurn,
  SUMMARY_REPLY,
  SwitchMarker,
  TABLE_REPLY,
  UserTurn,
  type ThreadState,
} from "./PrototypeThread";
import {
  CountPicker,
  DEFAULT_GUARDRAILS,
  GUARDRAIL_GROUPS,
  ModelPicker,
  VECTOR_STORE_GROUPS,
} from "./CleanSlatePickers";
import { CodeTab, CopyIconButton } from "./CleanSlateCode";
import { McpDialog } from "./McpDialog";

export type OpenControl = "model" | "mcp" | "stores" | "guardrails" | "params" | "key" | null;

const KEY_SOURCE_ITEMS = [
  { value: "session", label: "Current UI session" },
  { value: "custom", label: "Virtual key" },
];

type ModelNames = { previous: string; requested: string; served: string };

const SHORT_NAMES: ModelNames = {
  previous: "anthropic-haiku-4-5",
  requested: "anthropic-sonnet-5",
  served: "anthropic-sonnet-4-6",
};

const LONG_NAMES: ModelNames = {
  previous: "anthropic-haiku-4-5",
  requested: "bedrock-converse-us-anthropic-claude-sonnet-5-cross-region-prod",
  served: "vertex-ai-claude-sonnet-4-6-europe-west4-provisioned-throughput",
};

const NamesContext = createContext<ModelNames>(SHORT_NAMES);

function ModelName({ name, detail, className }: { name: string; detail?: string; className?: string }) {
  return (
    <HoverCard>
      <HoverCardTrigger
        delay={300}
        render={<span tabIndex={0} className={cn("min-w-0 cursor-default truncate font-mono", className)} />}
      >
        {name}
      </HoverCardTrigger>
      <HoverCardContent align="start" className="w-80">
        <div className="flex items-start gap-2">
          <div className="flex min-w-0 flex-1 flex-col gap-1">
            {detail && <span className="text-xs text-muted-foreground">{detail}</span>}
            <span className="font-mono text-sm wrap-anywhere">{name}</span>
          </div>
          <CopyIconButton value={name} label="Copy model name" />
        </div>
      </HoverCardContent>
    </HoverCard>
  );
}

function FallbackBadge() {
  return (
    <Badge variant="outline" className="border-warning/20 bg-warning/10 text-warning">
      <Split data-icon="inline-start" />
      Fallback
    </Badge>
  );
}

function EndpointMenu() {
  const [endpoint, setEndpoint] = useState("chat");
  const current = ENDPOINT_OPTIONS.find((option) => option.value === endpoint);
  return (
    <DropdownMenu>
      <DropdownMenuTrigger render={<InputGroupButton size="xs" className="shrink-0" />}>
        <span className="font-mono text-xs">{(current?.label ?? "/v1/chat/completions").replace(/^\/v1\//, "")}</span>
        <ChevronDown className="text-muted-foreground" />
      </DropdownMenuTrigger>
      <DropdownMenuContent align="start" className="w-64">
        <DropdownMenuLabel>Endpoint</DropdownMenuLabel>
        <DropdownMenuRadioGroup value={endpoint} onValueChange={setEndpoint}>
          {ENDPOINT_OPTIONS.map((option) => (
            <DropdownMenuRadioItem key={option.value} value={option.value}>
              {option.label}
            </DropdownMenuRadioItem>
          ))}
        </DropdownMenuRadioGroup>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

const TAG_OPTIONS = ["playground", "eval", "prod-canary", "team-support", "load-test"];

function TagPicker() {
  const anchor = useComboboxAnchor();
  return (
    <Combobox items={TAG_OPTIONS} multiple defaultValue={["playground"]}>
      <ComboboxChips ref={anchor} className="min-h-8 py-1">
        <ComboboxValue>
          {(values: string[]) => values.map((value) => <ComboboxChip key={value}>{value}</ComboboxChip>)}
        </ComboboxValue>
        <ComboboxChipsInput id="cs-tags" placeholder="Add a tag" />
      </ComboboxChips>
      <ComboboxContent anchor={anchor}>
        <ComboboxEmpty>No matching tags</ComboboxEmpty>
        <ComboboxList>
          {(item: string) => (
            <ComboboxItem key={item} value={item}>
              {item}
            </ComboboxItem>
          )}
        </ComboboxList>
      </ComboboxContent>
    </Combobox>
  );
}

function HeaderEditor() {
  const [rows, setRows] = useState<readonly (readonly [string, string])[]>([["anthropic-beta", "context-1m"]]);
  return (
    <div className="flex flex-col gap-2">
      {rows.map(([name, value], index) => (
        <ButtonGroup key={index} className="w-full">
          <Input aria-label="Header name" defaultValue={name} placeholder="Name" className="h-8 font-mono text-xs" />
          <Input aria-label="Header value" defaultValue={value} placeholder="Value" className="h-8 font-mono text-xs" />
          <Button
            variant="outline"
            size="icon-sm"
            aria-label={`Remove ${name || "header"}`}
            onClick={() => setRows(rows.filter((_, i) => i !== index))}
          >
            <Minus />
          </Button>
        </ButtonGroup>
      ))}
      <Button variant="ghost" size="sm" className="-ml-2 w-fit" onClick={() => setRows([...rows, ["", ""]])}>
        <Plus />
        Add header
      </Button>
    </div>
  );
}

function SettingsFields({ advancedOpen, children }: { advancedOpen: boolean; children?: ReactNode }) {
  const [source, setSource] = useState("session");
  const [temperature, setTemperature] = useState(1);
  return (
    <FieldGroup className="gap-5 p-4">
      <Field>
        <FieldLabel htmlFor="cs-key">Send as</FieldLabel>
        <Select items={KEY_SOURCE_ITEMS} value={source} onValueChange={(v) => setSource(v as string)}>
          <SelectTrigger id="cs-key" size="sm" className="w-full">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {KEY_SOURCE_ITEMS.map((item) => (
              <SelectItem key={item.value} value={item.value}>
                {item.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <FieldDescription>Pick a virtual key to test what it can access</FieldDescription>
      </Field>
      {children}
      <Field>
        <FieldLabel htmlFor="cs-tags">Tags</FieldLabel>
        <TagPicker />
      </Field>
      <FieldSet className="gap-3">
        <Field orientation="horizontal">
          <FieldLabel htmlFor="cs-stream" className="font-normal">
            Stream response
          </FieldLabel>
          <Switch id="cs-stream" defaultChecked />
        </Field>
        <Field orientation="horizontal">
          <FieldLabel htmlFor="cs-fallback" className="font-normal">
            Simulate a fallback
          </FieldLabel>
          <Switch id="cs-fallback" />
        </Field>
      </FieldSet>
      <Collapsible defaultOpen={advancedOpen}>
        <CollapsibleTrigger
          render={
            <Button
              variant="ghost"
              size="sm"
              className="-ml-2 w-fit text-muted-foreground aria-expanded:bg-transparent"
            />
          }
        >
          <ChevronRight className="transition-transform in-data-panel-open:rotate-90" />
          Advanced
        </CollapsibleTrigger>
        <CollapsibleContent>
          <FieldGroup className="gap-5 pt-3">
            <Field>
              <FieldLabel>Custom headers</FieldLabel>
              <HeaderEditor />
            </Field>
            <Field>
              <FieldLabel htmlFor="cs-base">Proxy base URL</FieldLabel>
              <Input id="cs-base" placeholder="Same as this dashboard" className="h-8" />
            </Field>
            <Field orientation="horizontal">
              <FieldLabel htmlFor="cs-temp" className="w-28 shrink-0 font-normal">
                Temperature
              </FieldLabel>
              <Slider
                id="cs-temp"
                min={0}
                max={2}
                step={0.1}
                value={[temperature]}
                onValueChange={(value) => setTemperature(Array.isArray(value) ? value[0] : value)}
              />
              <span className="w-8 shrink-0 text-right text-sm tabular-nums">{temperature.toFixed(1)}</span>
            </Field>
            <Field orientation="horizontal">
              <FieldLabel htmlFor="cs-max" className="w-28 shrink-0 font-normal">
                Max tokens
              </FieldLabel>
              <Input id="cs-max" type="number" placeholder="Default" className="h-8 w-24" />
            </Field>
          </FieldGroup>
        </CollapsibleContent>
      </Collapsible>
    </FieldGroup>
  );
}

function SettingsPopover({ open, advancedOpen }: { open: boolean; advancedOpen: boolean }) {
  return (
    <Popover defaultOpen={open}>
      <PopoverTrigger render={<InputGroupButton size="xs" />}>
        <SlidersHorizontal />
        Settings
      </PopoverTrigger>
      <PopoverContent align="end" side="top" className="max-h-(--available-height) w-96 overflow-y-auto p-0">
        <SettingsFields advancedOpen={advancedOpen} />
      </PopoverContent>
    </Popover>
  );
}

function SendButtons() {
  return (
    <>
      <InputGroupButton size="icon-xs" aria-label="Attach file">
        <Paperclip />
      </InputGroupButton>
      <InputGroupButton size="icon-xs" variant="default" aria-label="Send">
        <ArrowUp />
      </InputGroupButton>
    </>
  );
}

function ModelRow({ openControl }: { openControl: OpenControl }) {
  const names = useContext(NamesContext);
  return (
    <InputGroupAddon align="block-start" className="gap-1 border-b border-border pb-2">
      <ModelPicker initial={names.requested} open={openControl === "model"} />
      <EndpointMenu />
    </InputGroupAddon>
  );
}

const MESSAGE_FIELD = <InputGroupTextarea placeholder="Send a test request" rows={2} aria-label="Message" />;

function Composer({ openControl }: { openControl: OpenControl }) {
  return (
    <InputGroup>
      <ModelRow openControl={openControl} />
      {MESSAGE_FIELD}
      <InputGroupAddon align="block-end" className="gap-1">
        <McpDialog open={openControl === "mcp"} />
        <CountPicker
          label="Vector stores"
          icon={<Database />}
          groups={VECTOR_STORE_GROUPS}
          initial={["support-docs"]}
          open={openControl === "stores"}
        />
        <CountPicker
          label="Guardrails"
          icon={<Shield />}
          groups={GUARDRAIL_GROUPS}
          initial={DEFAULT_GUARDRAILS}
          open={openControl === "guardrails"}
        />
        <span className="ml-auto flex items-center gap-1">
          <SettingsPopover
            open={openControl === "params" || openControl === "key"}
            advancedOpen={openControl === "params"}
          />
          <SendButtons />
        </span>
      </InputGroupAddon>
    </InputGroup>
  );
}

type ReplyStats = {
  model: string;
  fellBack?: boolean;
  ttft: number;
  total: number;
  tokensIn: number;
  tokensOut: number;
  cachedTokens?: number;
  cost: string;
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

function ReplyMetrics({ stats }: { stats: ReplyStats }) {
  return (
    <div
      aria-label="Request metrics"
      className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-muted-foreground tabular-nums"
    >
      <Metric icon={<Timer />} value={`${stats.ttft.toFixed(2)}s`} label="to first token" />
      <Metric icon={<Clock />} value={`${stats.total.toFixed(2)}s`} label="total" />
      <Metric icon={<ArrowDownToLine />} value={String(stats.tokensIn)} label="in" />
      <Metric icon={<ArrowUpFromLine />} value={String(stats.tokensOut)} label="out" />
      {stats.cachedTokens ? <Metric icon={<DatabaseZap />} value={String(stats.cachedTokens)} label="cached" /> : null}
      <Metric icon={<Coins />} value={`$${stats.cost}`} />
    </div>
  );
}

function AssistantReply({ text, inspected, stats }: { text: string; inspected: boolean; stats: ReplyStats }) {
  return (
    <Message>
      <MessageContent className="text-sm">
        <MessageHeader className="gap-2 px-0 font-normal">
          <ModelName
            name={stats.model}
            detail={stats.fellBack ? "Served by, after 1 fallback" : "Served by"}
            className="max-w-96 text-xs text-muted-foreground"
          />
          {stats.fellBack && <FallbackBadge />}
        </MessageHeader>
        <Markdown>{text}</Markdown>
        <MessageFooter className="items-start gap-2 px-0 font-normal">
          <div className="min-w-0 flex-1">
            <ReplyMetrics stats={stats} />
          </div>
          <span className="flex shrink-0 items-center gap-0.5">
            <Tooltip>
              <TooltipTrigger render={<Button variant="ghost" size="icon-xs" aria-label="Copy reply" />}>
                <Copy />
              </TooltipTrigger>
              <TooltipContent>Copy reply</TooltipContent>
            </Tooltip>
            <Toggle
              size="sm"
              defaultPressed={inspected}
              aria-label="Inspect this request"
              className="h-6 gap-1 px-1.5 text-xs"
            >
              <ScanSearch />
              Inspect
            </Toggle>
          </span>
        </MessageFooter>
      </MessageContent>
    </Message>
  );
}

function Conversation({ state }: { state: ThreadState }) {
  const names = useContext(NamesContext);
  if (state === "empty") {
    return (
      <Empty className="flex-1">
        <EmptyHeader>
          <EmptyMedia variant="icon">
            <PlayCircle />
          </EmptyMedia>
          <EmptyTitle>Send a test request</EmptyTitle>
          <EmptyDescription>
            Pick a model and anything the request should go through, then inspect what the gateway did with it
          </EmptyDescription>
        </EmptyHeader>
      </Empty>
    );
  }
  return (
    <MessageScrollerProvider autoScroll defaultScrollPosition="end">
      <MessageScroller className="min-h-0 flex-1">
        <MessageScrollerViewport aria-label="Conversation" className="scroll-fade-b">
          <MessageScrollerContent className="mx-auto w-full max-w-3xl gap-6 px-6 py-6">
            <MessageScrollerItem messageId="u1">
              <UserTurn text="Give me a 3-row markdown table of LLM gateways and one sentence on each" />
            </MessageScrollerItem>
            <MessageScrollerItem messageId="a1">
              <AssistantReply
                text={TABLE_REPLY}
                inspected={false}
                stats={{ model: names.previous, ttft: 0.31, total: 1.42, tokensIn: 38, tokensOut: 112, cost: "0.0001" }}
              />
            </MessageScrollerItem>
            <MessageScrollerItem messageId="m1">
              <SwitchMarker model={names.requested} />
            </MessageScrollerItem>
            <MessageScrollerItem messageId="u2">
              <UserTurn text="What does this spec ask for?" file="team-budgets-spec.pdf" />
            </MessageScrollerItem>
            <MessageScrollerItem messageId="a2" scrollAnchor={state !== "streaming"}>
              {state === "streaming" ? (
                <StreamingTurn />
              ) : (
                <AssistantReply
                  text={SUMMARY_REPLY}
                  inspected
                  stats={{
                    model: names.served,
                    fellBack: true,
                    ttft: 0.62,
                    total: 2.8,
                    tokensIn: 1842,
                    tokensOut: 112,
                    cachedTokens: 1536,
                    cost: "0.0042",
                  }}
                />
              )}
            </MessageScrollerItem>
          </MessageScrollerContent>
        </MessageScrollerViewport>
        <MessageScrollerButton />
      </MessageScroller>
    </MessageScrollerProvider>
  );
}

const DEPLOYMENT_ID = "2147a42b69a7bc866ee542c0f9a30c8a45b8a770d19ae9a8aba8b07319191b44";
const CALL_ID = "7c1e9a52-3f4b-4d1e-9a0c-1b2e8f6d4a10";

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

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <Item size="sm" className="flex-col flex-nowrap items-start gap-0.5 p-0">
      <ItemDescription className="whitespace-nowrap">{label}</ItemDescription>
      <ItemTitle className="tabular-nums">{value}</ItemTitle>
    </Item>
  );
}

function StatGrid({ children }: { children: ReactNode }) {
  return <ItemGroup className="grid grid-cols-3 gap-x-4 gap-y-3 py-2 has-data-[size=sm]:gap-y-3">{children}</ItemGroup>;
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

function ProviderModel() {
  return (
    <HoverCard>
      <HoverCardTrigger
        delay={300}
        render={<span tabIndex={0} className="min-w-0 cursor-default truncate font-mono text-xs" />}
      >
        anthropic/claude-sonnet-4-6
      </HoverCardTrigger>
      <HoverCardContent align="end" className="w-80">
        <div className="flex items-start gap-2">
          <div className="flex min-w-0 flex-1 flex-col gap-1">
            <span className="text-xs text-muted-foreground">
              Deployment ID, the model entry on this proxy that ran it
            </span>
            <span className="font-mono text-xs wrap-anywhere">{DEPLOYMENT_ID}</span>
          </div>
          <CopyIconButton value={DEPLOYMENT_ID} label="Copy deployment ID" />
        </div>
      </HoverCardContent>
    </HoverCard>
  );
}

function Trace() {
  const names = useContext(NamesContext);
  return (
    <Accordion multiple defaultValue={["routing", "guardrails", "performance"]}>
      <TraceSection value="routing" title="Routing">
        <TraceRow title="Requested">
          <ModelName name={names.requested} detail="Requested model" className="text-xs" />
        </TraceRow>
        <TraceRow title="Served by">
          <FallbackBadge />
          <ModelName name={names.served} detail="Served by, after 1 fallback" className="text-xs" />
        </TraceRow>
        <TraceRow title="Provider model">
          <ProviderModel />
        </TraceRow>
        <TraceRow title="Call ID">
          <span className="min-w-0 truncate font-mono text-xs">{CALL_ID}</span>
          <CopyIconButton value={CALL_ID} label="Copy call ID" />
        </TraceRow>
        <StatGrid>
          <Stat label="Fallbacks" value="1" />
          <Stat label="Retries" value="0" />
          <Stat label="Overhead" value="14ms" />
        </StatGrid>
      </TraceSection>
      <TraceSection value="guardrails" title="Guardrails and policies">
        <TraceRow title="pii-mask" description="Guardrail">
          <Badge variant="secondary">Ran</Badge>
        </TraceRow>
        <TraceRow title="prompt-injection" description="Guardrail">
          <Badge variant="secondary">Ran</Badge>
        </TraceRow>
        <TraceRow title="pii-baseline" description="Policy, applies to all keys">
          <Badge variant="secondary">Ran</Badge>
        </TraceRow>
      </TraceSection>
      <TraceSection value="performance" title="Speed and cost">
        <StatGrid>
          <Stat label="First token" value="0.62s" />
          <Stat label="Total" value="2.80s" />
          <Stat label="Cost" value="$0.0042" />
          <Stat label="Tokens in" value="1842" />
          <Stat label="Tokens out" value="112" />
          <Stat label="Cached" value="1536" />
        </StatGrid>
      </TraceSection>
    </Accordion>
  );
}

function NothingYet() {
  return (
    <Empty className="flex-1">
      <EmptyHeader>
        <EmptyTitle className="text-sm">Nothing to inspect yet</EmptyTitle>
        <EmptyDescription>
          Each reply shows here with the deployment that served it, fallbacks, guardrails, timing, cost and the code to
          send it yourself
        </EmptyDescription>
      </EmptyHeader>
    </Empty>
  );
}

function Inspector({ state }: { state: ThreadState }) {
  return (
    <aside aria-label="Inspector" className="hidden w-96 shrink-0 flex-col border-l border-border xl:flex">
      {state !== "filled" ? (
        <NothingYet />
      ) : (
        <Tabs defaultValue="trace" className="min-h-0 flex-1 gap-0">
          <div className="flex h-12 shrink-0 items-center border-b border-border px-4">
            <TabsList className="w-full">
              <TabsTrigger value="trace">Trace</TabsTrigger>
              <TabsTrigger value="code">Code</TabsTrigger>
            </TabsList>
          </div>
          <ScrollArea className="min-h-0 flex-1">
            <TabsContent value="trace">
              <Trace />
            </TabsContent>
            <TabsContent value="code">
              <CodeTab />
            </TabsContent>
          </ScrollArea>
        </Tabs>
      )}
    </aside>
  );
}

export function PrototypeCleanSlate({
  state,
  openControl,
  longNames,
}: {
  state: ThreadState;
  openControl: OpenControl;
  longNames: boolean;
}) {
  return (
    <NamesContext.Provider value={longNames ? LONG_NAMES : SHORT_NAMES}>
      <div className="flex min-h-0 flex-1 flex-col gap-6">
        <PageHeader>
          <PageHeaderTitle>
            <PlayCircle />
            Playground
          </PageHeaderTitle>
          <PageHeaderDescription>
            Send test requests through the gateway with any model, key, guardrail or tool
          </PageHeaderDescription>
        </PageHeader>
        <div className="flex min-h-0 flex-1 flex-col">
          <PageTabs value="chat" className="flex-none">
            <PageTabsList>
              <PageTabsTrigger value="chat">Chat</PageTabsTrigger>
              <PageTabsTrigger value="compare">Compare</PageTabsTrigger>
              <PageTabsTrigger value="compliance">Compliance</PageTabsTrigger>
              <PageTabsTrigger value="system-one">
                <BetaBadge>Decisions</BetaBadge>
              </PageTabsTrigger>
              <PageTabsTrigger value="agent-builder">Agent builder</PageTabsTrigger>
            </PageTabsList>
          </PageTabs>
          <div className="flex min-h-0 flex-1">
            <div className="flex min-h-0 min-w-0 flex-1 flex-col">
              <div className="flex h-12 shrink-0 items-center gap-2 border-b border-border px-4">
                <span className="text-xs text-muted-foreground tabular-nums">
                  {state === "empty" ? "New chat" : "2 requests, $0.0043"}
                </span>
                <ButtonGroup className="ml-auto">
                  <Button variant="outline" size="sm">
                    <SquarePen />
                    New chat
                  </Button>
                  <DropdownMenu>
                    <DropdownMenuTrigger
                      render={<Button variant="outline" size="icon-sm" aria-label="More chat actions" />}
                    >
                      <ChevronDown />
                    </DropdownMenuTrigger>
                    <DropdownMenuContent align="end" className="w-64">
                      <DropdownMenuItem>
                        <SquarePen />
                        New chat, keep settings
                      </DropdownMenuItem>
                      <DropdownMenuItem>
                        <RotateCcw />
                        Reset chat and settings
                      </DropdownMenuItem>
                    </DropdownMenuContent>
                  </DropdownMenu>
                </ButtonGroup>
              </div>
              <Conversation state={state} />
              <div className="mx-auto w-full max-w-3xl px-6 pb-6">
                <Composer openControl={openControl} />
              </div>
            </div>
            <Inspector state={state} />
          </div>
        </div>
      </div>
    </NamesContext.Provider>
  );
}

export function parseOpenControl(value: string | null): OpenControl {
  const controls: readonly OpenControl[] = ["model", "mcp", "stores", "guardrails", "params", "key"];
  return controls.find((control) => control === value) ?? null;
}
