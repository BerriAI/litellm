"use client";

import React, { createContext, useContext, useState, useEffect } from "react";
import { getProxyBaseUrl } from "@/components/networking";

export type PluginMode = "ai-gateway" | "litellm-platform-plugin";

export interface PluginNavItem {
  key: string;
  label: string;
  icon?: string;
  path: string;
  badge?: boolean;
}

export interface Plugin {
  name: string;
  display_name: string;
  url: string;
  plugin_key?: string;
  nav_items: PluginNavItem[];
  capabilities: string[];
}

interface PluginModeContextValue {
  mode: PluginMode;
  setMode: (mode: PluginMode) => void;
  pluginKey: string | null;
  agentPlatformUrl: string;
  agentPlatformPath: string;
  setAgentPlatformPath: (path: string) => void;
}

const PluginModeContext = createContext<PluginModeContextValue>({
  mode: "ai-gateway",
  setMode: () => {},
  pluginKey: null,
  agentPlatformUrl: "",
  agentPlatformPath: "/sessions",
  setAgentPlatformPath: () => {},
});

const STORAGE_KEY = "litellm_plugin_mode";

export function PluginModeProvider({ children }: { children: React.ReactNode }) {
  const [mode, setModeState] = useState<PluginMode>("ai-gateway");
  const [pluginKey, setPluginKey] = useState<string | null>(null);
  const [agentPlatformUrl, setAgentPlatformUrl] = useState<string>("");
  const [agentPlatformPath, setAgentPlatformPath] = useState<string>("/sessions");

  useEffect(() => {
    const stored = localStorage.getItem(STORAGE_KEY) as PluginMode | null;
    if (stored === "litellm-platform-plugin" || stored === "ai-gateway") {
      setModeState(stored);
    }

    // Fetch plugin registry to get plugin_key and url
    const base = getProxyBaseUrl() ?? "";
    fetch(`${base}/api/plugins`, {
      credentials: "include",
      headers: { Authorization: `Bearer ${document.cookie.match(/token=([^;]+)/)?.[1] ?? ""}` },
    })
      .then((r) => r.json())
      .then((plugins: Plugin[]) => {
        const lap = plugins.find((p) => p.name === "litellm-platform-plugin");
        if (lap) {
          setAgentPlatformUrl(lap.url);
          if (lap.plugin_key) setPluginKey(lap.plugin_key);
        }
      })
      .catch(() => {});
  }, []);

  const setMode = (m: PluginMode) => {
    setModeState(m);
    localStorage.setItem(STORAGE_KEY, m);
  };

  return (
    <PluginModeContext.Provider value={{ mode, setMode, pluginKey, agentPlatformUrl, agentPlatformPath, setAgentPlatformPath }}>
      {children}
    </PluginModeContext.Provider>
  );
}

export function usePluginMode() {
  return useContext(PluginModeContext);
}
