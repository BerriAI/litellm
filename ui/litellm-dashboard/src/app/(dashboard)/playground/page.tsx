"use client";

import { useState, useEffect } from "react";
import AgentBuilderView from "@/components/playground/chat_ui/AgentBuilderView";
import ChatUI from "@/components/playground/chat_ui/ChatUI";
import CompareUI from "@/components/playground/compareUI/CompareUI";
import ComplianceUI from "@/components/playground/complianceUI/ComplianceUI";
import { TabGroup, TabList, Tab, TabPanels, TabPanel } from "@tremor/react";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { fetchProxySettings } from "@/utils/proxyUtils";
import { useTranslation } from "react-i18next";

interface ProxySettings {
  PROXY_BASE_URL?: string;
  LITELLM_UI_API_DOC_BASE_URL?: string | null;
}

export default function PlaygroundPage() {
  const { t } = useTranslation();
  const { accessToken, userRole, userId, disabledPersonalKeyCreation, token } = useAuthorized();
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

  return (
    <div className="h-full w-full flex flex-col">
      <TabGroup className="w-full" style={{ flex: 1, minHeight: 0, display: "flex", flexDirection: "column" }}>
        <TabList className="mb-0">
          <Tab>{t("pages.playgroundPage.tabChat")}</Tab>
          <Tab>{t("pages.playgroundPage.tabCompare")}</Tab>
          <Tab>{t("pages.playgroundPage.tabCompliance")}</Tab>
          <Tab>{t("pages.playgroundPage.tabAgentBuilder")}</Tab>
        </TabList>
        <TabPanels className="h-full">
          <TabPanel className="h-full">
            <ChatUI
              accessToken={accessToken}
              token={token}
              userRole={userRole}
              userID={userId}
              disabledPersonalKeyCreation={disabledPersonalKeyCreation}
              proxySettings={proxySettings}
            />
          </TabPanel>
          <TabPanel className="h-full">
            <CompareUI accessToken={accessToken} disabledPersonalKeyCreation={disabledPersonalKeyCreation} />
          </TabPanel>
          <TabPanel className="h-full">
            <ComplianceUI accessToken={accessToken} disabledPersonalKeyCreation={disabledPersonalKeyCreation} />
          </TabPanel>
          <TabPanel className="h-full">
            <AgentBuilderView
              accessToken={accessToken}
              token={token}
              userID={userId}
              userRole={userRole}
              disabledPersonalKeyCreation={disabledPersonalKeyCreation}
              proxySettings={proxySettings}
              customProxyBaseUrl={proxySettings?.LITELLM_UI_API_DOC_BASE_URL ?? proxySettings?.PROXY_BASE_URL}
            />
          </TabPanel>
        </TabPanels>
      </TabGroup>
    </div>
  );
}
