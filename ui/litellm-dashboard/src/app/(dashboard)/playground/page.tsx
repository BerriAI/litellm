"use client";

import { useState, useEffect } from "react";
import AgentBuilderView from "@/components/playground/chat_ui/AgentBuilderView";
import ChatUI from "@/components/playground/chat_ui/ChatUI";
import CompareUI from "@/components/playground/compareUI/CompareUI";
import ComplianceUI from "@/components/playground/complianceUI/ComplianceUI";
import { TabGroup, TabList, Tab, TabPanels, TabPanel } from "@tremor/react";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { fetchProxySettings } from "@/utils/proxyUtils";
import { MessageOutlined, CloseOutlined } from "@ant-design/icons";

interface ProxySettings {
  PROXY_BASE_URL?: string;
  LITELLM_UI_API_DOC_BASE_URL?: string | null;
}

export default function PlaygroundPage() {
  const { accessToken, userRole, userId, disabledPersonalKeyCreation, token } = useAuthorized();
  const [proxySettings, setProxySettings] = useState<ProxySettings | undefined>(undefined);
  const [chatBannerDismissed, setChatBannerDismissed] = useState(false);

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
      {!chatBannerDismissed && (
        <div style={{
          display: "flex",
          alignItems: "center",
          gap: 16,
          padding: "10px 20px",
          background: "#f0f9ff",
          borderBottom: "1px solid #bae6fd",
          flexShrink: 0,
        }}>
          <span style={{
            fontSize: 10,
            fontWeight: 700,
            color: "#fff",
            background: "#0ea5e9",
            borderRadius: 4,
            padding: "2px 7px",
            letterSpacing: "0.08em",
            textTransform: "uppercase",
            flexShrink: 0,
            lineHeight: "18px",
          }}>
            New
          </span>
          <span style={{ flex: 1, color: "#0c4a6e", fontSize: 13.5, lineHeight: 1.5 }}>
            <strong>Chat UI</strong>
            {" "}— a ChatGPT-like interface for your users to chat with AI models and MCP tools. Share it with your team.
          </span>
          <a
            href="/chat"
            target="_blank"
            rel="noopener noreferrer"
            style={{
              display: "inline-flex",
              alignItems: "center",
              gap: 5,
              padding: "5px 14px",
              borderRadius: 6,
              background: "#0ea5e9",
              color: "#fff",
              fontSize: 12.5,
              fontWeight: 600,
              textDecoration: "none",
              whiteSpace: "nowrap",
              flexShrink: 0,
            }}
          >
            Open Chat UI →
          </a>
          <button
            onClick={() => setChatBannerDismissed(true)}
            style={{ background: "none", border: "none", cursor: "pointer", color: "#64748b", padding: 4, flexShrink: 0, lineHeight: 1 }}
            aria-label="Dismiss"
          >
            <CloseOutlined style={{ fontSize: 13 }} />
          </button>
        </div>
      )}
    <TabGroup className="w-full" style={{ flex: 1, minHeight: 0, display: "flex", flexDirection: "column" }}>
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
    </div>
  );
}
