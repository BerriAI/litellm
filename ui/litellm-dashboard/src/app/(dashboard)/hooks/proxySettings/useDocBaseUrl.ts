"use client";

import { getProxyBaseUrl } from "@/components/networking";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import useProxySettings from "@/app/(dashboard)/hooks/proxySettings/useProxySettings";

export function resolveDocBaseUrl(docBaseUrl: string | null | undefined, fallback: string): string {
  return docBaseUrl && docBaseUrl.trim() ? docBaseUrl : fallback;
}

export default function useDocBaseUrl(): string {
  const { accessToken } = useAuthorized();
  const { LITELLM_UI_API_DOC_BASE_URL } = useProxySettings(accessToken);
  return resolveDocBaseUrl(LITELLM_UI_API_DOC_BASE_URL, getProxyBaseUrl());
}
