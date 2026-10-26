"use client";

import { useState, useEffect } from "react";
import { PlayCircle } from "lucide-react";
import AgentBuilderView from "@/app/(dashboard)/playground/components/chat_ui/AgentBuilderView";
import { PlaygroundChat, type ProxySettings } from "@/app/(dashboard)/playground/components/chat/PlaygroundChat";
import CompareUI from "@/app/(dashboard)/playground/components/compareUI/CompareUI";
import ComplianceUI from "@/app/(dashboard)/playground/components/complianceUI/ComplianceUI";
import SystemOneUI from "@/app/(dashboard)/playground/components/systemOneUI/SystemOneUI";
import BetaBadge from "@/components/BetaBadge";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { fetchProxySettings } from "@/utils/proxyUtils";
import { useUrlTab } from "@/hooks/useUrlTab";
import { Page, PageTabs, PageTabsContent, PageTabsList, PageTabsTrigger } from "@/components/shared/Page";
import { PageHeader, PageHeaderDescription, PageHeaderTitle } from "@/components/shared/PageHeader";

const PLAYGROUND_TABS = ["chat", "compare", "compliance", "system-one", "agent-builder"] as const;

const TAB_TRIGGER = "h-full group-data-horizontal/tabs:after:-bottom-0.5";

export default function PlaygroundPage() {
  const { accessToken, userRole, userId, disabledPersonalKeyCreation, token, isViewOnly } = useAuthorized();
  const [proxySettings, setProxySettings] = useState<ProxySettings | undefined>(undefined);
  const [activeTab, setActiveTab] = useUrlTab(PLAYGROUND_TABS, "chat");

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

  return (
    <Page className="h-full overflow-hidden pb-0 sm:pb-0">
      <PageHeader>
        <PageHeaderTitle>
          <PlayCircle />
          Playground
        </PageHeaderTitle>
        <PageHeaderDescription>
          Send test requests through the gateway with any model, key, guardrail or tool
        </PageHeaderDescription>
      </PageHeader>
      <PageTabs value={activeTab} onValueChange={setActiveTab} className="gap-0">
        <PageTabsList>
          <PageTabsTrigger value="chat" className={TAB_TRIGGER}>
            Chat
          </PageTabsTrigger>
          <PageTabsTrigger value="compare" className={TAB_TRIGGER}>
            Compare
          </PageTabsTrigger>
          <PageTabsTrigger value="compliance" className={TAB_TRIGGER}>
            Compliance
          </PageTabsTrigger>
          <PageTabsTrigger value="system-one" className={TAB_TRIGGER}>
            <BetaBadge>Decisions</BetaBadge>
          </PageTabsTrigger>
          <PageTabsTrigger value="agent-builder" className={TAB_TRIGGER}>
            Agent builder
          </PageTabsTrigger>
        </PageTabsList>
        <PageTabsContent value="chat" className="overflow-hidden data-hidden:hidden" keepMounted>
          <PlaygroundChat
            accessToken={accessToken}
            disabledPersonalKeyCreation={disabledPersonalKeyCreation}
            proxySettings={proxySettings}
          />
        </PageTabsContent>
        <PageTabsContent value="compare" className="overflow-y-auto pt-4 data-hidden:hidden" keepMounted>
          <CompareUI accessToken={accessToken} disabledPersonalKeyCreation={disabledPersonalKeyCreation} />
        </PageTabsContent>
        <PageTabsContent value="compliance" className="overflow-y-auto pt-4 data-hidden:hidden" keepMounted>
          <ComplianceUI accessToken={accessToken} disabledPersonalKeyCreation={disabledPersonalKeyCreation} />
        </PageTabsContent>
        <PageTabsContent value="system-one" className="overflow-y-auto pt-4 data-hidden:hidden" keepMounted>
          <SystemOneUI accessToken={accessToken} disabledPersonalKeyCreation={disabledPersonalKeyCreation} />
        </PageTabsContent>
        <PageTabsContent value="agent-builder" className="overflow-y-auto pt-4 data-hidden:hidden" keepMounted>
          <AgentBuilderView
            accessToken={accessToken}
            token={token}
            userID={userId}
            userRole={userRole}
            disabledPersonalKeyCreation={disabledPersonalKeyCreation}
            proxySettings={proxySettings}
            customProxyBaseUrl={proxySettings?.LITELLM_UI_API_DOC_BASE_URL ?? proxySettings?.PROXY_BASE_URL}
          />
        </PageTabsContent>
      </PageTabs>
    </Page>
  );
}
