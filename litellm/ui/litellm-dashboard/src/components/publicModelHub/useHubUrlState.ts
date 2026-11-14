"use client";

import { createSerializer, parseAsString, parseAsStringLiteral, useQueryStates } from "nuqs";
import { usePathname, useSearchParams } from "next/navigation";
import { useCallback, useEffect } from "react";

import { recordUiEvent } from "@/lib/telemetry/uiEvents";
import { routeSegmentForPathname } from "@/utils/uiHref";

export const HUB_TABS = ["models", "skills"] as const;
export type HubTab = (typeof HUB_TABS)[number];

const OPEN_PARAM = { open: parseAsString };
const HUB_PARAMS = {
  tab: parseAsStringLiteral([...HUB_TABS, "agents", "mcp"] as const).withDefault("models"),
  ...OPEN_PARAM,
};
const serializeOpen = createSerializer(OPEN_PARAM);

export function useHubUrlState() {
  const [state, setState] = useQueryStates(HUB_PARAMS);
  const page = routeSegmentForPathname(usePathname() ?? "") || "home";
  const retired = state.tab === "agents" || state.tab === "mcp";
  const tab: HubTab = state.tab === "skills" ? "skills" : "models";
  useEffect(() => {
    if (retired) void setState({ tab: null, open: null });
  }, [retired, setState]);
  const setOpen = useCallback((key: string | null) => void setState({ open: key }), [setState]);
  const setTab = useCallback(
    (next: HubTab) => {
      void recordUiEvent({ page, action: "click", target: `tab=${next}` });
      void setState({ tab: next, open: null });
    },
    [page, setState],
  );
  return { tab, setTab, open: retired ? null : state.open || null, setOpen };
}

export function useOpenHref() {
  const params = useSearchParams();
  return useCallback((key: string) => serializeOpen(params ?? new URLSearchParams(), { open: key }), [params]);
}
