"use client";

import { useState, useEffect } from "react";
import AgentBuilderView from "@/app/(dashboard)/playground/components/chat_ui/AgentBuilderView";
import ChatUI from "@/app/(dashboard)/playground/components/chat_ui/ChatUI";
import CompareUI from "@/app/(dashboard)/playground/components/compareUI/CompareUI";
import ComplianceUI from "@/app/(dashboard)/playground/components/complianceUI/ComplianceUI";
import SystemOneUI from "@/app/(dashboard)/playground/components/systemOneUI/SystemOneUI";
import BetaBadge from "@/components/BetaBadge";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { fetchProxySettings } from "@/utils/proxyUtils";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useUrlTab } from "@/hooks/useUrlTab";
import { parseAsString, useQueryState } from "nuqs";
import { Code2, Eraser, PlayCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { Page } from "@/components/shared/Page";
import { PageHeader, PageHeaderControls, PageHeaderDescription, PageHeaderTitle } from "@/components/shared/PageHeader";
import {
  PrototypeOptionA,
  PrototypeOptionB,
} from "@/app/(dashboard)/playground/components/chat_prototype/PrototypeOptions";
import {
  PrototypeCleanSlate,
  parseOpenControl,
} from "@/app/(dashboard)/playground/components/chat_prototype/PrototypeCleanSlate";
import { PrototypeHeaderOption } from "@/app/(dashboard)/playground/components/chat_prototype/PrototypeHeaderOptions";

const PLAYGROUND_TABS = ["chat", "compare", "compliance", "system-one", "agent-builder"] as const;

interface ProxySettings {
  PROXY_BASE_URL?: string;
  LITELLM_UI_API_DOC_BASE_URL?: string | null;
}

export default function PlaygroundPage() {
  const { accessToken, userRole, userId, disabledPersonalKeyCreation, token, isViewOnly } = useAuthorized();
  const [proxySettings, setProxySettings] = useState<ProxySettings | undefined>(undefined);
  const [activeTab, setActiveTab] = useUrlTab(PLAYGROUND_TABS, "chat");
  const [proto] = useQueryState("proto", parseAsString);
  const [sheet] = useQueryState("sheet", parseAsString);
  const [thread] = useQueryState("thread", parseAsString);
  const [hdr] = useQueryState("hdr", parseAsString);
  const [open] = useQueryState("open", parseAsString);
  const [names] = useQueryState("names", parseAsString);
  const threadState = thread === "empty" || thread === "streaming" ? thread : "filled";

  useEffect(() => {
    const initializeProxySettings = async () => {
      if (accessToken) {
        const settings = await fetchProxySettings(accessToken);
        if (settings) {
          setProxySettings({
            PROXY_BASE_URL: settings.PROXY_BASE_URL,
            LITELLM_UI_API_DOC_BASE_URL: settings.LITELLM_UI_API_DOC_BASE_URL,
          });
        }
      }
    };

    initializeProxySettings();
  }, [accessToken]);

  if (isViewOnly) {
    return (
      <div className="flex h-full w-full flex-col items-center justify-center gap-2 p-8 text-center">
        <h1 className="text-2xl font-semibold">Access Denied</h1>
        <p className="text-muted-foreground">
          Your role does not have access to the Playground. Ask your proxy admin for access to test models.
        </p>
      </div>
    );
  }

  if (proto === "clean") {
    return (
      <Page className="h-full overflow-hidden pb-0 sm:pb-0">
        <PrototypeCleanSlate state={threadState} openControl={parseOpenControl(open)} longNames={names === "long"} />
      </Page>
    );
  }

  if (proto === "a" && (hdr === "1" || hdr === "2" || hdr === "3")) {
    return (
      <Page className="h-full overflow-hidden pb-0 sm:pb-0">
        <PrototypeHeaderOption variant={hdr} threadState={threadState} />
      </Page>
    );
  }

  if (proto === "a" || proto === "b") {
    return (
      <Page className="h-full gap-0 overflow-hidden pb-0 sm:pb-0">
        <PageHeader>
          <PageHeaderTitle>
            <PlayCircle />
            Playground
          </PageHeaderTitle>
          <PageHeaderDescription>
            Send test requests through the gateway with any model, key, guardrail or tool
          </PageHeaderDescription>
          <PageHeaderControls className="border-b border-border">
            <Button variant="outline" size="sm">
              <Code2 />
              Get code
            </Button>
            <Button variant="outline" size="sm">
              <Eraser />
              Clear chat
            </Button>
            <Separator orientation="vertical" className="mx-2 h-5" />
            <Tabs value="chat">
              <TabsList variant="line" className="h-9 gap-4 p-0">
                <TabsTrigger value="chat" className="flex-none px-0">
                  Chat
                </TabsTrigger>
                <TabsTrigger value="compare" className="flex-none px-0">
                  Compare
                </TabsTrigger>
                <TabsTrigger value="compliance" className="flex-none px-0">
                  Compliance
                </TabsTrigger>
                <TabsTrigger value="system-one" className="flex-none px-0">
                  <BetaBadge>Decisions</BetaBadge>
                </TabsTrigger>
                <TabsTrigger value="agent-builder" className="flex-none px-0">
                  Agent builder
                </TabsTrigger>
              </TabsList>
            </Tabs>
          </PageHeaderControls>
        </PageHeader>
        {proto === "a" ? (
          <PrototypeOptionA threadState={threadState} />
        ) : (
          <PrototypeOptionB sheetOpen={sheet === "1"} threadState={threadState} />
        )}
      </Page>
    );
  }

  return (
    <div className="flex h-full min-h-0 w-full min-w-0 flex-col overflow-hidden">
      <Tabs
        value={activeTab}
        onValueChange={setActiveTab}
        className="flex min-h-0 min-w-0 flex-1 flex-col gap-0 overflow-hidden"
      >
        <TabsList variant="line" className="w-full shrink-0 justify-start overflow-x-auto pb-1">
          <TabsTrigger value="chat" className="flex-none">
            Chat
          </TabsTrigger>
          <TabsTrigger value="compare" className="flex-none">
            Compare
          </TabsTrigger>
          <TabsTrigger value="compliance" className="flex-none">
            Compliance
          </TabsTrigger>
          <TabsTrigger value="system-one" className="flex-none">
            <BetaBadge>Decisions</BetaBadge>
          </TabsTrigger>
          <TabsTrigger value="agent-builder" className="flex-none">
            Agent Builder (Experimental)
          </TabsTrigger>
        </TabsList>
        <TabsContent
          value="chat"
          className="mt-0 h-full min-h-0 min-w-0 overflow-hidden data-hidden:hidden"
          keepMounted
        >
          <ChatUI
            accessToken={accessToken}
            token={token}
            userRole={userRole}
            userID={userId}
            disabledPersonalKeyCreation={disabledPersonalKeyCreation}
            proxySettings={proxySettings}
          />
        </TabsContent>
        <TabsContent value="compare" className="mt-0 h-full data-hidden:hidden" keepMounted>
          <CompareUI accessToken={accessToken} disabledPersonalKeyCreation={disabledPersonalKeyCreation} />
        </TabsContent>
        <TabsContent value="compliance" className="mt-0 h-full data-hidden:hidden" keepMounted>
          <ComplianceUI accessToken={accessToken} disabledPersonalKeyCreation={disabledPersonalKeyCreation} />
        </TabsContent>
        <TabsContent value="system-one" className="mt-0 min-h-0 data-hidden:hidden" keepMounted>
          <SystemOneUI accessToken={accessToken} disabledPersonalKeyCreation={disabledPersonalKeyCreation} />
        </TabsContent>
        <TabsContent value="agent-builder" className="mt-0 h-full data-hidden:hidden" keepMounted>
          <AgentBuilderView
            accessToken={accessToken}
            token={token}
            userID={userId}
            userRole={userRole}
            disabledPersonalKeyCreation={disabledPersonalKeyCreation}
            proxySettings={proxySettings}
            customProxyBaseUrl={proxySettings?.LITELLM_UI_API_DOC_BASE_URL ?? proxySettings?.PROXY_BASE_URL}
          />
        </TabsContent>
      </Tabs>
    </div>
  );
}
