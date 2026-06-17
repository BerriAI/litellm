"use client";

import React, { createContext, useContext, useState, useEffect } from "react";

export type PluginMode = "ai-gateway" | "agent-control-plane";

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
  nav_items: PluginNavItem[];
  capabilities: string[];
}

interface PluginModeContextValue {
  mode: PluginMode;
  setMode: (mode: PluginMode) => void;
  agentPlatformUrl: string;
  setAgentPlatformUrl: (url: string) => void;
  agentPlatformPath: string;
  setAgentPlatformPath: (path: string) => void;
}

const PluginModeContext = createContext<PluginModeContextValue>({
  mode: "ai-gateway",
  setMode: () => {},
  agentPlatformUrl: "",
  setAgentPlatformUrl: () => {},
  agentPlatformPath: "/sessions",
  setAgentPlatformPath: () => {},
});

const STORAGE_KEY = "litellm_plugin_mode";
const AGENT_URL_KEY = "litellm_agent_platform_url";

export function PluginModeProvider({ children }: { children: React.ReactNode }) {
  const [mode, setModeState] = useState<PluginMode>("ai-gateway");
  const [agentPlatformUrl, setAgentPlatformUrlState] = useState<string>("");
  const [agentPlatformPath, setAgentPlatformPath] = useState<string>("/sessions");

  useEffect(() => {
    const stored = localStorage.getItem(STORAGE_KEY) as PluginMode | null;
    if (stored === "agent-control-plane" || stored === "ai-gateway") {
      setModeState(stored);
    }
    const storedUrl = localStorage.getItem(AGENT_URL_KEY);
    if (storedUrl) {
      setAgentPlatformUrlState(storedUrl);
    } else {
      // Default: same host, port 3210 (agent platform default dev port)
      const defaultUrl = typeof window !== "undefined"
        ? `${window.location.protocol}//${window.location.hostname}:3210`
        : "http://localhost:3210";
      setAgentPlatformUrlState(defaultUrl);
    }
  }, []);

  const setMode = (m: PluginMode) => {
    setModeState(m);
    localStorage.setItem(STORAGE_KEY, m);
  };

  const setAgentPlatformUrl = (url: string) => {
    setAgentPlatformUrlState(url);
    localStorage.setItem(AGENT_URL_KEY, url);
  };

  return (
    <PluginModeContext.Provider value={{ mode, setMode, agentPlatformUrl, setAgentPlatformUrl, agentPlatformPath, setAgentPlatformPath }}>
      {children}
    </PluginModeContext.Provider>
  );
}

export function usePluginMode() {
  return useContext(PluginModeContext);
}
