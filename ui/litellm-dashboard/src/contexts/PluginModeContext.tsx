"use client";

import React, { createContext, useContext, useState, useEffect } from "react";
import { createApiClient } from "@/lib/http/client";
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
const pluginApiClient = createApiClient({ getBaseUrl: () => getProxyBaseUrl() ?? "" });

function readStoredMode(): PluginMode {
  if (typeof window === "undefined") return "ai-gateway";
  const stored = localStorage.getItem(STORAGE_KEY) as PluginMode | null;
  return stored === "litellm-platform-plugin" || stored === "ai-gateway" ? stored : "ai-gateway";
}

export function PluginModeProvider({ children }: { children: React.ReactNode }) {
  // Lazy initializer reads localStorage once — no setState in effect
  const [mode, setModeState] = useState<PluginMode>(readStoredMode);
  const [pluginKey, setPluginKey] = useState<string | null>(null);
  const [agentPlatformUrl, setAgentPlatformUrl] = useState<string>("");
  const [agentPlatformPath, setAgentPlatformPath] = useState<string>("/sessions");

  useEffect(() => {
    const token = document.cookie.match(/token=([^;]+)/)?.[1] ?? localStorage.getItem("token") ?? "";
    pluginApiClient
      .get("/api/plugins", { accessToken: token })
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
    <PluginModeContext.Provider
      value={{
        mode,
        setMode,
        pluginKey,
        agentPlatformUrl,
        agentPlatformPath,
        setAgentPlatformPath,
      }}
    >
      {children}
    </PluginModeContext.Provider>
  );
}

export function usePluginMode() {
  return useContext(PluginModeContext);
}
