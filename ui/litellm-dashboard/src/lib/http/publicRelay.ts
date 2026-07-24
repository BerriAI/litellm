"use client";

import { getProxyBaseUrl } from "@/components/networking";

export type Money = {
  currency: "USD";
  amount_micros: string;
  display: string;
};

export type Account = {
  account_id: string;
  user_id: string;
  email: string;
  status: "ACTIVE" | "FROZEN" | "CLOSED";
  created_at: string;
};

export type ApiKey = {
  key_id: string;
  alias: string | null;
  key: string | null;
  created_at: string | null;
  log_content: boolean;
};

export type SessionResult = {
  account: Account;
  csrf_token: string;
  default_key: ApiKey | null;
};

export async function relayFetch<T>(
  path: string,
  init: RequestInit = {},
  csrfToken?: string,
): Promise<{ data: T; response: Response }> {
  const headers = new Headers(init.headers);
  if (init.body) {
    headers.set("content-type", "application/json");
  }
  if (csrfToken) {
    headers.set("x-csrf-token", csrfToken);
  }
  const response = await fetch(`${getProxyBaseUrl()}${path}`, {
    ...init,
    headers,
    credentials: "include",
  });
  const data = (await response.json()) as T & { detail?: unknown };
  if (!response.ok) {
    throw new Error(errorMessage(data.detail));
  }
  return { data, response };
}

function errorMessage(detail: unknown): string {
  if (typeof detail === "string") {
    return detail;
  }
  if (detail && typeof detail === "object" && "error" in detail) {
    const error = (detail as { error: unknown }).error;
    if (typeof error === "string") {
      return error;
    }
    if (error && typeof error === "object" && "message" in error) {
      return String((error as { message: unknown }).message);
    }
  }
  return "The request could not be completed";
}
