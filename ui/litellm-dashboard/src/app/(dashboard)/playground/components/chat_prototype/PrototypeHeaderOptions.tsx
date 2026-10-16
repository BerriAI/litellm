"use client";

import type { ReactNode } from "react";
import { Code2, Ellipsis, PlayCircle, SquarePen } from "lucide-react";
import CopyButton from "@/components/shared/CopyButton";
import { PageTabs, PageTabsList, PageTabsTrigger } from "@/components/shared/Page";
import { PageHeader, PageHeaderDescription, PageHeaderTitle } from "@/components/shared/PageHeader";
import BetaBadge from "@/components/BetaBadge";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { ResizableHandle, ResizablePanel, ResizablePanelGroup } from "@/components/ui/Resizable";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { SettingsPanel } from "./PrototypeOptions";
import { PROTOTYPE_MODEL, PrototypeThread, type ThreadState } from "./PrototypeThread";

const PYTHON_SNIPPET = `from openai import OpenAI

client = OpenAI(api_key="sk-...", base_url="http://localhost:4000")

response = client.chat.completions.create(
    model="${PROTOTYPE_MODEL}",
    messages=[{"role": "user", "content": "Give me a 3-row markdown table of LLM gateways"}],
    temperature=1.0,
    max_tokens=2048,
    extra_body={"guardrails": ["pii-mask"]},
)`;

const CURL_SNIPPET = `curl http://localhost:4000/v1/chat/completions \\
  -H "Authorization: Bearer sk-..." \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "${PROTOTYPE_MODEL}",
    "messages": [{"role": "user", "content": "Give me a 3-row markdown table of LLM gateways"}],
    "guardrails": ["pii-mask"]
  }'`;

function HeaderTitle({ actions }: { actions?: ReactNode }) {
  return (
    <PageHeader className="flex items-start justify-between gap-4">
      <div>
        <PageHeaderTitle>
          <PlayCircle />
          Playground
        </PageHeaderTitle>
        <PageHeaderDescription>
          Send test requests through the gateway with any model, key, guardrail or tool
        </PageHeaderDescription>
      </div>
      {actions}
    </PageHeader>
  );
}

function PlaygroundTabs() {
  return (
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
  );
}

function IconAction({ label, children }: { label: string; children: ReactNode }) {
  return (
    <Tooltip>
      <TooltipTrigger render={<Button variant="ghost" size="icon-sm" aria-label={label} />}>{children}</TooltipTrigger>
      <TooltipContent>{label}</TooltipContent>
    </Tooltip>
  );
}

function ThreadToolbar({ withCode }: { withCode: boolean }) {
  return (
    <div className="flex h-12 shrink-0 items-center gap-1 border-b border-border px-4">
      <span className="text-sm font-medium">Conversation</span>
      <span className="ml-2 text-xs text-muted-foreground">2 requests, $0.0043</span>
      <span className="ml-auto flex items-center gap-1">
        {withCode && (
          <IconAction label="Get code">
            <Code2 />
          </IconAction>
        )}
        <IconAction label="New chat">
          <SquarePen />
        </IconAction>
      </span>
    </div>
  );
}

function CodePanel() {
  return (
    <Tabs defaultValue="python" className="flex h-full min-h-0 flex-col gap-0">
      <div className="flex h-12 shrink-0 items-center gap-2 border-b border-border px-4">
        <span className="text-sm font-medium">Code</span>
        <TabsList className="ml-auto h-7">
          <TabsTrigger value="python" className="px-2 text-xs">
            Python
          </TabsTrigger>
          <TabsTrigger value="curl" className="px-2 text-xs">
            cURL
          </TabsTrigger>
        </TabsList>
      </div>
      {[
        ["python", PYTHON_SNIPPET],
        ["curl", CURL_SNIPPET],
      ].map(([value, code]) => (
        <TabsContent key={value} value={value} className="relative min-h-0 flex-1">
          <ScrollArea className="h-full">
            <pre className="p-4 font-mono text-xs leading-5 whitespace-pre-wrap text-foreground">{code}</pre>
          </ScrollArea>
          <div className="absolute top-2 right-2">
            <CopyButton value={code} label="Copy code" />
          </div>
        </TabsContent>
      ))}
      <p className="border-t border-border px-4 py-3 text-xs text-muted-foreground">
        Updates as you change the model and settings on the left
      </p>
    </Tabs>
  );
}

export type HeaderVariant = "1" | "2" | "3";

export function PrototypeHeaderOption({ variant, threadState }: { variant: HeaderVariant; threadState: ThreadState }) {
  const menu = (
    <DropdownMenu>
      <DropdownMenuTrigger render={<Button variant="outline" size="icon-sm" aria-label="More actions" />}>
        <Ellipsis />
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end">
        <DropdownMenuItem>
          <Code2 />
          Get code
        </DropdownMenuItem>
        <DropdownMenuItem>
          <SquarePen />
          New chat
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-6">
      <HeaderTitle actions={variant === "2" ? menu : undefined} />
      <div className="flex min-h-0 flex-1 flex-col">
        <PlaygroundTabs />
        {variant === "3" ? (
          <ResizablePanelGroup orientation="horizontal" className="min-h-0 flex-1">
            <ResizablePanel defaultSize={288} minSize={240} maxSize={420}>
              <SettingsPanel />
            </ResizablePanel>
            <ResizableHandle />
            <ResizablePanel minSize={420}>
              <div className="flex h-full min-h-0 flex-col">
                <ThreadToolbar withCode={false} />
                <PrototypeThread state={threadState} />
              </div>
            </ResizablePanel>
            <ResizableHandle />
            <ResizablePanel defaultSize={360} minSize={280} maxSize={560}>
              <CodePanel />
            </ResizablePanel>
          </ResizablePanelGroup>
        ) : (
          <div className="flex min-h-0 flex-1">
            <div className="w-72 shrink-0 border-r border-border">
              <SettingsPanel />
            </div>
            <div className="flex min-h-0 min-w-0 flex-1 flex-col">
              {variant === "1" && <ThreadToolbar withCode />}
              <PrototypeThread state={threadState} />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
