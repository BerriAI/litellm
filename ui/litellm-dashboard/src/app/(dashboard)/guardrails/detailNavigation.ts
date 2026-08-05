import { useSearchParams } from "next/navigation";
import { useCallback } from "react";

import { migratedHref } from "@/utils/migratedPages";
import { navigateWithParams } from "../navigateWithParams";

export const GUARDRAIL_PARAM = "guardrail";
export const GUARDRAIL_TAB_PARAM = "guardrail_tab";

export type GuardrailDetailTab = "overview" | "settings";

export interface GuardrailDetailRouting {
  guardrailId: string | null;
  tab: GuardrailDetailTab;
  openGuardrail: (id: string) => void;
  close: () => void;
}

/** Same-origin href that opens a guardrail's detail view, e.g. "/ui/guardrails?guardrail=abc&guardrail_tab=settings". */
export function guardrailDetailHref(guardrailId: string, tab: GuardrailDetailTab = "overview"): string {
  const params = new URLSearchParams(
    tab === "settings"
      ? { [GUARDRAIL_PARAM]: guardrailId, [GUARDRAIL_TAB_PARAM]: tab }
      : { [GUARDRAIL_PARAM]: guardrailId },
  );
  return `${migratedHref("guardrails")}?${params.toString()}`;
}

export function useGuardrailDetailRouting(): GuardrailDetailRouting {
  const searchParams = useSearchParams();

  const openGuardrail = useCallback((id: string) => {
    navigateWithParams((params) => {
      params.set(GUARDRAIL_PARAM, id);
      params.delete(GUARDRAIL_TAB_PARAM);
    });
  }, []);

  const close = useCallback(() => {
    navigateWithParams((params) => {
      params.delete(GUARDRAIL_PARAM);
      params.delete(GUARDRAIL_TAB_PARAM);
    });
  }, []);

  return {
    guardrailId: searchParams?.get(GUARDRAIL_PARAM) ?? null,
    tab: searchParams?.get(GUARDRAIL_TAB_PARAM) === "settings" ? "settings" : "overview",
    openGuardrail,
    close,
  };
}
