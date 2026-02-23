"use client";

import { useState, useEffect } from "react";
import AgentBuilderView from "@/components/playground/chat_ui/AgentBuilderView";
import ChatUI from "@/components/playground/chat_ui/ChatUI";
import CompareUI from "@/components/playground/compareUI/CompareUI";
import ComplianceUI from "@/components/playground/complianceUI/ComplianceUI";
import { TabGroup, TabList, Tab, TabPanels, TabPanel } from "@tremor/react";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { fetchProxySettings } from "@/utils/proxyUtils";

interface ProxySettings {
  PROXY_BASE_URL?: string;
  LITELLM_UI_API_DOC_BASE_URL?: string | null;
}

export default function PlaygroundPage() {
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
    <TabGroup className="h-full w-full">
      <TabList className="mb-0">
        <Tab>Chat</Tab>
        <Tab>Compare</Tab>
        <Tab>Compliance</Tab>
        <Tab>Agent Builder (Experimental)</Tab>
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
  );
}
