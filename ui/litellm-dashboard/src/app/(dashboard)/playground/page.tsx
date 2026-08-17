"use client";

import { useState, useEffect } from "react";
import AgentBuilderView from "@/app/(dashboard)/playground/components/chat_ui/AgentBuilderView";
import ChatUI from "@/app/(dashboard)/playground/components/chat_ui/ChatUI";
import CompareUI from "@/app/(dashboard)/playground/components/compareUI/CompareUI";
import ComplianceUI from "@/app/(dashboard)/playground/components/complianceUI/ComplianceUI";
import { DeprecationBanner } from "@/components/DeprecationBanner";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { fetchProxySettings } from "@/utils/proxyUtils";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useTranslation } from "react-i18next";

interface ProxySettings {
  PROXY_BASE_URL?: string;
  LITELLM_UI_API_DOC_BASE_URL?: string | null;
}

export default function PlaygroundPage() {
  const { t } = useTranslation();
  const { accessToken, userRole, userId, disabledPersonalKeyCreation, token, isViewOnly } = useAuthorized();
  const [proxySettings, setProxySettings] = useState<ProxySettings | undefined>(undefined);

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
        <h1 className="text-2xl font-semibold">{t("playground.accessDenied.title")}</h1>
        <p className="text-muted-foreground">{t("playground.accessDenied.description")}</p>
      </div>
    );
  }

  return (
    <div className="flex h-full min-h-0 w-full min-w-0 flex-col overflow-hidden">
      <Tabs defaultValue="chat" className="flex min-h-0 min-w-0 flex-1 flex-col gap-0 overflow-hidden">
        <TabsList variant="line" className="w-full shrink-0 justify-start overflow-x-auto pb-1">
          <TabsTrigger value="chat" className="flex-none">
            {t("playground.tabs.chat")}
          </TabsTrigger>
          <TabsTrigger value="compare" className="flex-none">
            {t("playground.tabs.compare")}
          </TabsTrigger>
          <TabsTrigger value="compliance" className="flex-none">
            {t("playground.tabs.compliance")}
          </TabsTrigger>
          <TabsTrigger value="agent-builder" className="flex-none">
            {t("playground.tabs.agentBuilder")}
          </TabsTrigger>
        </TabsList>
        <TabsContent value="chat" className="mt-0 h-full min-h-0 min-w-0 overflow-hidden data-hidden:hidden">
          <ChatUI
            accessToken={accessToken}
            token={token}
            userRole={userRole}
            userID={userId}
            disabledPersonalKeyCreation={disabledPersonalKeyCreation}
            proxySettings={proxySettings}
          />
        </TabsContent>
        <TabsContent value="compare" className="mt-0 h-full data-hidden:hidden">
          <CompareUI accessToken={accessToken} disabledPersonalKeyCreation={disabledPersonalKeyCreation} />
        </TabsContent>
        <TabsContent value="compliance" className="mt-0 h-full data-hidden:hidden">
          <ComplianceUI accessToken={accessToken} disabledPersonalKeyCreation={disabledPersonalKeyCreation} />
        </TabsContent>
        <TabsContent value="agent-builder" className="mt-0 h-full data-hidden:hidden">
          <DeprecationBanner featureName="The Playground's Agent Builder" />
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
