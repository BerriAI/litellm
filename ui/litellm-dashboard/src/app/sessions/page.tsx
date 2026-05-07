"use client";

import React, { useEffect } from "react";
import { useRouter } from "next/navigation";

const DEFAULT_PROXY = "http://localhost:4000";
const DEFAULT_KEY = "sk-1234";

const getProxyBase = (): string =>
  (typeof window !== "undefined" &&
    localStorage.getItem("LITELLM_PROXY_URL")) ||
  DEFAULT_PROXY;

const getApiKey = (): string =>
  (typeof window !== "undefined" && localStorage.getItem("LITELLM_API_KEY")) ||
  DEFAULT_KEY;

export default function SessionsIndexPage() {
  const router = useRouter();

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const res = await fetch(`${getProxyBase()}/v2/sessions?limit=1`, {
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${getApiKey()}`,
          },
        });
        if (!res.ok || cancelled) return;
        const data: { data: { id: string }[] } = await res.json();
        const first = data.data?.[0];
        if (first && !cancelled) {
          router.replace(`/sessions/${first.id}`);
        }
      } catch {
        // silent
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [router]);

  return (
    <div
      style={{
        padding: 24,
        color: "var(--text-muted)",
        fontSize: 13,
      }}
    >
      Loading sessions…
    </div>
  );
}
