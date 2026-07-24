"use client";

import { useState } from "react";
import { CacheHealthTab } from "@/app/(dashboard)/caching/_components/cache_health";
import NotificationsManager from "@/components/molecules/notifications_manager";
import { cachingHealthCheckCall } from "@/components/networking";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

interface CacheHealthResponse {
  status?: string;
  cache_type?: string;
  ping_response?: boolean;
  set_cache_response?: string;
  litellm_cache_params?: string;
  error?: { message?: string; type?: string; param?: string; code?: string };
}

const parseHealthCheckError = (message: string): CacheHealthResponse["error"] => {
  try {
    const parsed: unknown = JSON.parse(message);
    if (parsed && typeof parsed === "object") {
      const outer = parsed as Record<string, unknown>;
      const inner = outer.error;
      const source = inner && typeof inner === "object" ? (inner as Record<string, unknown>) : outer;
      const read = (field: string) => (typeof source[field] === "string" ? (source[field] as string) : undefined);
      return {
        message: read("message") ?? message,
        type: read("type"),
        param: read("param"),
        code: read("code"),
      };
    }
  } catch {
    // message is not JSON; fall through to the raw string
  }
  return { message };
};

export default function CacheHealthPage() {
  const { accessToken } = useAuthorized();
  const [healthCheckResponse, setHealthCheckResponse] = useState<CacheHealthResponse | string>("");

  const runCachingHealthCheck = async () => {
    try {
      NotificationsManager.info("Running cache health check...");
      setHealthCheckResponse("");
      const response = await cachingHealthCheckCall(accessToken ?? "");
      setHealthCheckResponse(response);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unknown error occurred";
      setHealthCheckResponse({ error: parseHealthCheckError(message) });
    }
  };

  return (
    <CacheHealthTab
      accessToken={accessToken}
      healthCheckResponse={healthCheckResponse}
      runCachingHealthCheck={runCachingHealthCheck}
    />
  );
}
