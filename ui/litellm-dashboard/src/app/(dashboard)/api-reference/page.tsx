"use client";

import APIReferenceView from "@/app/(dashboard)/api-reference/APIReferenceView";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { fetchProxySettings } from "@/utils/proxyUtils";
import { useState, useEffect } from "react";

interface ProxySettings {
  PROXY_BASE_URL?: string;
  LITELLM_UI_API_DOC_BASE_URL?: string | null;
}

const APIReferencePage = () => {
  const { accessToken } = useAuthorized();
  const [proxySettings, setProxySettings] = useState<ProxySettings>({});

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

  return <APIReferenceView proxySettings={proxySettings} />;
};

export default APIReferencePage;
