"use client";

import React, { createContext, useContext, useState, useEffect, useCallback, ReactNode } from "react";
import { getProxyBaseUrl, setProxyBaseUrl, getGlobalLitellmHeaderName } from "@/components/networking";
import { getLocalStorageItem, setLocalStorageItem } from "@/utils/localStorageUtils";

const CONNECTIONS_KEY = "litellm_proxy_connections";
const ACTIVE_ID_KEY = "litellm_active_connection_id";

export interface ProxyConnection {
  id: string;
  name: string;
  url: string;
  apiKey: string;
  isDefault: boolean;
}

interface TestConnectionResult {
  ok: boolean;
  version?: string;
  error?: string;
}

interface ProxyConnectionContextType {
  connections: ProxyConnection[];
  activeConnection: ProxyConnection | null;
  addConnection: (conn: Omit<ProxyConnection, "id" | "isDefault">) => void;
  updateConnection: (id: string, updates: Partial<Pick<ProxyConnection, "name" | "url" | "apiKey">>) => void;
  removeConnection: (id: string) => void;
  switchConnection: (id: string) => void;
  testConnection: (url: string, apiKey: string) => Promise<TestConnectionResult>;
  isRemoteProxy: boolean;
}

const defaultContextValue: ProxyConnectionContextType = {
  connections: [],
  activeConnection: null,
  addConnection: () => {},
  updateConnection: () => {},
  removeConnection: () => {},
  switchConnection: () => {},
  testConnection: async () => ({ ok: false, error: "No provider" }),
  isRemoteProxy: false,
};

const ProxyConnectionContext = createContext<ProxyConnectionContextType>(defaultContextValue);

export const useProxyConnection = () => {
  return useContext(ProxyConnectionContext);
};

function generateId(): string {
  if (typeof crypto !== "undefined" && crypto.randomUUID) {
    return crypto.randomUUID();
  }
  return Math.random().toString(36).substring(2) + Date.now().toString(36);
}

function loadConnections(): ProxyConnection[] {
  const raw = getLocalStorageItem(CONNECTIONS_KEY);
  if (!raw) return [];
  try {
    return JSON.parse(raw);
  } catch {
    return [];
  }
}

function saveConnections(connections: ProxyConnection[]) {
  setLocalStorageItem(CONNECTIONS_KEY, JSON.stringify(connections));
}

function loadActiveId(): string | null {
  return getLocalStorageItem(ACTIVE_ID_KEY);
}

function saveActiveId(id: string) {
  setLocalStorageItem(ACTIVE_ID_KEY, id);
}

function ensureDefaultConnection(connections: ProxyConnection[]): ProxyConnection[] {
  const hasDefault = connections.some((c) => c.isDefault);
  if (hasDefault) return connections;

  const defaultUrl = getProxyBaseUrl();
  const defaultConn: ProxyConnection = {
    id: "default",
    name: "Default",
    url: defaultUrl,
    apiKey: "",
    isDefault: true,
  };
  return [defaultConn, ...connections];
}

interface ProxyConnectionProviderProps {
  children: ReactNode;
}

export const ProxyConnectionProvider: React.FC<ProxyConnectionProviderProps> = ({ children }) => {
  const [connections, setConnections] = useState<ProxyConnection[]>(() => {
    const loaded = loadConnections();
    return ensureDefaultConnection(loaded);
  });

  const [activeId, setActiveId] = useState<string>(() => {
    const savedId = loadActiveId();
    if (savedId) return savedId;
    const defaultConn = connections.find((c) => c.isDefault);
    return defaultConn?.id ?? "default";
  });

  const activeConnection = connections.find((c) => c.id === activeId) ?? connections.find((c) => c.isDefault) ?? null;
  const isRemoteProxy = activeConnection !== null && !activeConnection.isDefault;

  // On mount, set the proxyBaseUrl for non-default connections
  useEffect(() => {
    if (activeConnection && !activeConnection.isDefault) {
      setProxyBaseUrl(activeConnection.url);
    }
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Persist connections to localStorage whenever they change
  useEffect(() => {
    saveConnections(connections);
  }, [connections]);

  const addConnection = useCallback((conn: Omit<ProxyConnection, "id" | "isDefault">) => {
    const newConn: ProxyConnection = {
      ...conn,
      id: generateId(),
      isDefault: false,
    };
    setConnections((prev) => {
      const updated = [...prev, newConn];
      saveConnections(updated);
      return updated;
    });
  }, []);

  const updateConnection = useCallback((id: string, updates: Partial<Pick<ProxyConnection, "name" | "url" | "apiKey">>) => {
    setConnections((prev) => {
      const updated = prev.map((c) => (c.id === id ? { ...c, ...updates } : c));
      saveConnections(updated);
      return updated;
    });
  }, []);

  const removeConnection = useCallback((id: string) => {
    setConnections((prev) => {
      const conn = prev.find((c) => c.id === id);
      if (!conn || conn.isDefault) return prev;
      const updated = prev.filter((c) => c.id !== id);
      saveConnections(updated);
      return updated;
    });

    // If removing the active connection, switch back to default
    if (id === activeId) {
      const defaultConn = connections.find((c) => c.isDefault);
      if (defaultConn) {
        saveActiveId(defaultConn.id);
        setProxyBaseUrl(defaultConn.url);
        window.location.reload();
      }
    }
  }, [activeId, connections]);

  const switchConnection = useCallback((id: string) => {
    if (id === activeId) return;
    const target = connections.find((c) => c.id === id);
    if (!target) return;

    saveActiveId(target.id);

    if (target.isDefault) {
      // Switching back to default — clear the override so getProxyBaseUrl() falls back
      setProxyBaseUrl(null);
    } else {
      setProxyBaseUrl(target.url);
    }

    window.location.reload();
  }, [activeId, connections]);

  const testConnection = useCallback(async (url: string, apiKey: string): Promise<TestConnectionResult> => {
    try {
      const cleanUrl = url.replace(/\/+$/, "");
      const response = await fetch(`${cleanUrl}/health/readiness`, {
        method: "GET",
        headers: {
          [getGlobalLitellmHeaderName()]: `Bearer ${apiKey}`,
          "Content-Type": "application/json",
        },
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        return { ok: false, error: errorData?.error || `HTTP ${response.status}` };
      }

      const data = await response.json();
      return { ok: true, version: data?.litellm_version };
    } catch (error: unknown) {
      const message = error instanceof Error ? error.message : "Connection failed";
      if (message.includes("Failed to fetch") || message.includes("NetworkError")) {
        return {
          ok: false,
          error: `Connection failed. Ensure the proxy is running and CORS is configured to allow requests from ${window.location.origin}`,
        };
      }
      return { ok: false, error: message };
    }
  }, []);

  return (
    <ProxyConnectionContext.Provider
      value={{
        connections,
        activeConnection,
        addConnection,
        updateConnection,
        removeConnection,
        switchConnection,
        testConnection,
        isRemoteProxy,
      }}
    >
      {children}
    </ProxyConnectionContext.Provider>
  );
};
