"use client";

import React, { createContext, useContext, useState, useEffect } from "react";
import { createApiClient } from "@/lib/http/client";
import { getProxyBaseUrl } from "@/components/networking";
import { hasCapability } from "@/utils/capabilities";

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
}

const PluginModeContext = createContext<PluginModeContextValue>({
  mode: "ai-gateway",
  setMode: () => {},
  plugins: [],
  activePlugin: null,
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
  /**
   * Pass the current user role from the app's auth context. `/api/plugins` is a
   * proxy-admin route, so a role without that capability never fetches; the
   * provider then behaves exactly as it does for an empty plugin list.
   */
  userRole: string | null;
}

export function PluginModeProvider({ children, accessToken, userRole }: PluginModeProviderProps) {
  const [mode, setModeState] = useState<PluginMode>(readStoredMode);
  const [plugins, setPlugins] = useState<Plugin[]>([]);
  const [loaded, setLoaded] = useState(false);
  const canViewPlugins = hasCapability(userRole, "viewPlugins");

  useEffect(() => {
    // Re-fetch whenever the auth token changes (handles login/logout cycles)
    if (!accessToken || !canViewPlugins) return;
    pluginApiClient
      .get("/api/plugins", { accessToken })
      .then((data: Plugin[]) => {
        setPlugins(Array.isArray(data) ? data : []);
      })
      .catch(() => {})
      // Mark loaded even on failure so a stored plugin mode still falls back to
      // ai-gateway; otherwise a failed fetch would strand the user on a blank
      // plugin view with no switcher to escape.
      .finally(() => setLoaded(true));
  }, [accessToken, canViewPlugins]);

  // A role that can't read plugins settles without ever fetching, so a stored
  // plugin mode still falls back below. Gated on the role having resolved so an
  // admin doesn't flash to ai-gateway while auth is still loading.
  const pluginListSettled = loaded || (Boolean(userRole) && !canViewPlugins);

  // Once the plugin list has settled, fall back to ai-gateway if the persisted
  // mode is no longer registered — including when the list came back empty (all
  // plugins removed). Derived rather than setState-in-effect to avoid cascading
  // renders.
  const effectiveMode =
    mode !== "ai-gateway" && pluginListSettled && !plugins.some((p) => p.name === mode) ? "ai-gateway" : mode;

  const setMode = (m: PluginMode) => {
    setModeState(m);
    localStorage.setItem(STORAGE_KEY, m);
  };

  const activePlugin = plugins.find((p) => p.name === effectiveMode) ?? null;

  return (
    <PluginModeContext.Provider value={{ mode: effectiveMode, setMode, plugins, activePlugin }}>
      {children}
    </PluginModeContext.Provider>
  );
}

export function usePluginMode() {
  return useContext(PluginModeContext);
}
