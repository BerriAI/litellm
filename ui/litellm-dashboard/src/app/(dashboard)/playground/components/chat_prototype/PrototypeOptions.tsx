"use client";

import type { ReactNode } from "react";
import { ChevronRight, Plug, Settings2, Shield, X } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle, SheetTrigger } from "@/components/ui/sheet";
import { Separator } from "@/components/ui/separator";
import {
  EndpointField,
  ModelField,
  ParameterFields,
  RequestFields,
  SafetyFields,
  ToolFields,
} from "./PrototypeSettings";
import { PrototypeThread, type ThreadState } from "./PrototypeThread";

function SettingsGroup({
  title,
  summary,
  defaultOpen = false,
  children,
}: {
  title: string;
  summary?: ReactNode;
  defaultOpen?: boolean;
  children: ReactNode;
}) {
  return (
    <Collapsible defaultOpen={defaultOpen} className="border-b border-border">
      <CollapsibleTrigger className="group flex w-full items-center gap-2 px-4 py-3 text-sm font-semibold hover:bg-muted/50">
        <ChevronRight className="size-4 text-muted-foreground transition-transform group-data-panel-open:rotate-90" />
        {title}
        <span className="ml-auto text-xs font-normal text-muted-foreground">{summary}</span>
      </CollapsibleTrigger>
      <CollapsibleContent className="px-4 pb-4">{children}</CollapsibleContent>
    </Collapsible>
  );
}

export function SettingsPanel() {
  return (
    <ScrollArea className="h-full">
      <SettingsGroup title="Model" defaultOpen>
        <div className="flex flex-col gap-4">
          <EndpointField />
          <ModelField />
          <ParameterFields />
        </div>
      </SettingsGroup>
      <SettingsGroup title="Tools and context" summary={<Badge variant="outline">1 set</Badge>}>
        <ToolFields />
      </SettingsGroup>
      <SettingsGroup title="Guardrails and policies" summary={<Badge variant="outline">1 set</Badge>}>
        <SafetyFields />
      </SettingsGroup>
      <SettingsGroup title="Request" summary="UI session">
        <RequestFields />
      </SettingsGroup>
    </ScrollArea>
  );
}

export function PrototypeOptionA({ threadState }: { threadState: ThreadState }) {
  return (
    <div className="flex min-h-0 flex-1 border-t border-border">
      <aside className="w-72 shrink-0 border-r border-border">
        <SettingsPanel />
      </aside>
      <PrototypeThread state={threadState} />
    </div>
  );
}

function ActiveChip({ icon, kind, value }: { icon: ReactNode; kind: string; value: string }) {
  return (
    <Button variant="outline" size="sm" className="gap-1.5 pr-1.5">
      {icon}
      <span className="font-normal text-muted-foreground">{kind}</span>
      {value}
      <X className="text-muted-foreground" aria-label={`Remove ${value}`} />
    </Button>
  );
}

export function PrototypeOptionB({ sheetOpen, threadState }: { sheetOpen: boolean; threadState: ThreadState }) {
  return (
    <div className="flex min-h-0 flex-1 flex-col border-t border-border">
      <div className="flex flex-wrap items-center gap-2 border-b border-border px-6 py-3">
        <div className="w-56">
          <EndpointField label={false} />
        </div>
        <div className="w-56">
          <ModelField label={false} />
        </div>
        <Separator orientation="vertical" className="mx-1 h-6" />
        <ActiveChip icon={<Shield />} kind="Guardrail" value="pii-mask" />
        <ActiveChip icon={<Plug />} kind="MCP" value="github" />
        <Sheet defaultOpen={sheetOpen}>
          <SheetTrigger render={<Button variant="outline" size="sm" className="ml-auto" />}>
            <Settings2 />
            Settings
            <Badge className="ml-0.5 h-4 min-w-4 px-1">2</Badge>
          </SheetTrigger>
          <SheetContent side="right" className="w-96 sm:max-w-96">
            <SheetHeader>
              <SheetTitle>Request settings</SheetTitle>
              <SheetDescription>Applied to every message you send from this tab</SheetDescription>
            </SheetHeader>
            <ScrollArea className="min-h-0 flex-1">
              <div className="flex flex-col gap-6 px-4 pb-6">
                <ParameterFields />
                <Separator />
                <SafetyFields />
                <Separator />
                <ToolFields />
                <Separator />
                <RequestFields />
              </div>
            </ScrollArea>
          </SheetContent>
        </Sheet>
      </div>
      <PrototypeThread state={threadState} />
    </div>
  );
}
