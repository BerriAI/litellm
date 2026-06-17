"use client";

import React, { createContext, useContext, useState, useEffect } from "react";
import { createApiClient } from "@/lib/http/client";
import { getProxyBaseUrl } from "@/components/networking";

export type PluginMode = "ai-gateway" | string; // "ai-gateway" or a registered plugin name

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
  nav_items?: PluginNavItem[];
  capabilities?: string[];
}

interface PluginModeContextValue {
  mode: PluginMode;
  setMode: (mode: PluginMode) => void;
  plugins: Plugin[];
  activePlugin: Plugin | null;
  agentPlatformPath: string;
  setAgentPlatformPath: (path: string) => void;
}

const PluginModeContext = createContext<PluginModeContextValue>({
  mode: "ai-gateway",
  setMode: () => {},
  plugins: [],
  activePlugin: null,
  agentPlatformPath: "/sessions",
  setAgentPlatformPath: () => {},
});

const STORAGE_KEY = "litellm_plugin_mode";
const pluginApiClient = createApiClient({ getBaseUrl: () => getProxyBaseUrl() ?? "" });

function readStoredMode(): PluginMode {
  if (typeof window === "undefined") return "ai-gateway";
  return localStorage.getItem(STORAGE_KEY) ?? "ai-gateway";
}

interface PluginModeProviderProps {
  children: React.ReactNode;
  /** Pass the current access token from the app's auth context. */
  accessToken?: string | null;
}

export function PluginModeProvider({ children, accessToken }: PluginModeProviderProps) {
  const [mode, setModeState] = useState<PluginMode>(readStoredMode);
  const [plugins, setPlugins] = useState<Plugin[]>([]);
  const [agentPlatformPath, setAgentPlatformPath] = useState<string>("/sessions");

  useEffect(() => {
    // Re-fetch whenever the auth token changes (handles login/logout cycles)
    if (!accessToken) return;
    pluginApiClient
      .get("/api/plugins", { accessToken })
      .then((data: Plugin[]) => {
        setPlugins(Array.isArray(data) ? data : []);
      })
      .catch(() => {});
  }, [accessToken]);

  // If the persisted mode is no longer registered, fall back to ai-gateway
  useEffect(() => {
    if (mode !== "ai-gateway" && plugins.length > 0) {
      const stillRegistered = plugins.some((p) => p.name === mode);
      if (!stillRegistered) setModeState("ai-gateway");
    }
  }, [plugins, mode]);

  const setMode = (m: PluginMode) => {
    setModeState(m);
    localStorage.setItem(STORAGE_KEY, m);
  };

  const activePlugin = plugins.find((p) => p.name === mode) ?? null;

  return (
    <PluginModeContext.Provider
      value={{ mode, setMode, plugins, activePlugin, agentPlatformPath, setAgentPlatformPath }}
    >
      {children}
    </PluginModeContext.Provider>
  );
}

export function usePluginMode() {
  return useContext(PluginModeContext);
}
