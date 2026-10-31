"use client";

import { createSerializer, parseAsString, useQueryState } from "nuqs";
import { useSearchParams } from "next/navigation";
import { useCallback } from "react";

import { useUrlTab } from "@/hooks/useUrlTab";

export const HUB_TABS = ["models", "agents", "mcp", "skills"] as const;
export type HubTab = (typeof HUB_TABS)[number];

const OPEN_PARAM = { open: parseAsString };
const serializeOpen = createSerializer(OPEN_PARAM);

export function useHubUrlState() {
  const [tab, setUrlTab] = useUrlTab(HUB_TABS, "models");
  const [open, setOpenParam] = useQueryState("open", OPEN_PARAM.open);
  const setOpen = useCallback((key: string | null) => void setOpenParam(key), [setOpenParam]);
  const setTab = useCallback(
    (next: HubTab) => {
      setUrlTab(next);
      setOpen(null);
    },
    [setUrlTab, setOpen],
  );
  return { tab, setTab, open: open || null, setOpen };
}

export function useOpenHref() {
  const params = useSearchParams();
  return useCallback((key: string) => serializeOpen(params ?? new URLSearchParams(), { open: key }), [params]);
}
