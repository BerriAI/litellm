"use client";

import ChatUI from "@/components/playground/chat_ui/ChatUI";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { useState, useEffect } from "react";
import { fetchProxySettings } from "@/utils/proxyUtils";

interface ProxySettings {
  PROXY_BASE_URL?: string;
  LITELLM_UI_API_DOC_BASE_URL?: string | null;
}

const TestKeyPage = () => {
  const { token, accessToken, userRole, userId, disabledPersonalKeyCreation } = useAuthorized();
  const [proxySettings, setProxySettings] = useState<ProxySettings | undefined>(undefined);

  useEffect(() => {
    const initializeProxySettings = async () => {
      if (accessToken) {
        const settings = await fetchProxySettings(accessToken);
        if (settings) {
          setProxySettings({
            PROXY_BASE_URL: settings.PROXY_BASE_URL || undefined,
            LITELLM_UI_API_DOC_BASE_URL: settings.LITELLM_UI_API_DOC_BASE_URL,
          });
        }
      }
    };

    initializeProxySettings();
  }, [accessToken]);

  return (
    <ChatUI
      accessToken={accessToken}
      token={token}
      userRole={userRole}
      userID={userId}
      disabledPersonalKeyCreation={disabledPersonalKeyCreation}
      proxySettings={proxySettings}
    />
  );
};

export default TestKeyPage;
