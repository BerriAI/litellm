import { useCallback, useEffect, useState } from "react";

import { UsageOption } from "../components/UsageViewSelect/UsageViewSelect";
import { replaceWindowSearchParams } from "@/utils/replaceWindowSearchParams";

export const USAGE_VIEW_PARAM = "view";
export const USAGE_TEAM_PARAM = "team";
export const USAGE_KEY_PARAM = "key";

export const USAGE_OPTIONS: readonly UsageOption[] = [
  "global",
  "my-usage",
  "my-budgets",
  "organization",
  "team",
  "customer",
  "tag",
  "agent",
  "user",
  "user-agent-activity",
] as const;

const USAGE_OPTION_SET: ReadonlySet<string> = new Set(USAGE_OPTIONS);

export function parseUsageView(raw: string | null | undefined): UsageOption {
  if (raw && USAGE_OPTION_SET.has(raw)) {
    return raw as UsageOption;
  }
  return "global";
}

export type UsageSearchParamsPatch = {
  view?: UsageOption | null;
  team?: string | null;
  key?: string | null;
};

export function buildUsageSearchParams(
  currentSearch: string,
  patch: UsageSearchParamsPatch,
): URLSearchParams {
  const normalized = currentSearch.startsWith("?") ? currentSearch.slice(1) : currentSearch;
  const next = new URLSearchParams(normalized);

  if ("view" in patch) {
    if (!patch.view) {
      next.delete(USAGE_VIEW_PARAM);
    } else {
      next.set(USAGE_VIEW_PARAM, patch.view);
    }
  }

  if ("team" in patch) {
    if (!patch.team) {
      next.delete(USAGE_TEAM_PARAM);
    } else {
      next.set(USAGE_TEAM_PARAM, patch.team);
    }
  }

  if ("key" in patch) {
    if (!patch.key) {
      next.delete(USAGE_KEY_PARAM);
    } else {
      next.set(USAGE_KEY_PARAM, patch.key);
    }
  }

  return next;
}

export function usageHref(pathname: string, searchParams: URLSearchParams): string {
  const qs = searchParams.toString();
  return qs ? `${pathname}?${qs}` : pathname;
}

function readUsageFromWindow(): {
  view: UsageOption;
  teamId: string | null;
  keyId: string | null;
} {
  if (typeof window === "undefined") {
    return { view: "global", teamId: null, keyId: null };
  }
  const params = new URLSearchParams(window.location.search);
  return {
    view: parseUsageView(params.get(USAGE_VIEW_PARAM)),
    teamId: params.get(USAGE_TEAM_PARAM),
    keyId: params.get(USAGE_KEY_PARAM),
  };
}

function applyUsagePatchToParams(params: URLSearchParams, patch: UsageSearchParamsPatch): void {
  if ("view" in patch) {
    if (!patch.view) {
      params.delete(USAGE_VIEW_PARAM);
    } else {
      // Always write view, including global. Omitting view=global collapses the URL
      // back to bare /ui/usage and can snap the dropdown back to a prior view.
      params.set(USAGE_VIEW_PARAM, patch.view);
    }
  }
  if ("team" in patch) {
    if (!patch.team) {
      params.delete(USAGE_TEAM_PARAM);
    } else {
      params.set(USAGE_TEAM_PARAM, patch.team);
    }
  }
  if ("key" in patch) {
    if (!patch.key) {
      params.delete(USAGE_KEY_PARAM);
    } else {
      params.set(USAGE_KEY_PARAM, patch.key);
    }
  }
}

/**
 * Sync Usage page UI state to URL search params so refresh/share keeps the same view,
 * My Budgets team, and open key detail from Top Keys.
 *
 * Local React state is the UI source of truth; the URL is updated via history.replaceState
 * so soft-nav under /ui keeps working (Next router.replace does not).
 */
export function useUsagePageSearchParams() {
  const [view, setView] = useState<UsageOption>(() => readUsageFromWindow().view);
  const [teamId, setTeamId] = useState<string | null>(() => readUsageFromWindow().teamId);
  const [keyId, setKeyId] = useState<string | null>(() => readUsageFromWindow().keyId);

  useEffect(() => {
    const syncFromWindow = () => {
      const next = readUsageFromWindow();
      setView(next.view);
      setTeamId(next.teamId);
      setKeyId(next.keyId);
    };
    window.addEventListener("popstate", syncFromWindow);
    return () => window.removeEventListener("popstate", syncFromWindow);
  }, []);

  const updateSearchParams = useCallback((patch: UsageSearchParamsPatch) => {
    if ("view" in patch) {
      setView(parseUsageView(patch.view));
    }
    if ("team" in patch) {
      setTeamId(patch.team ?? null);
    }
    if ("key" in patch) {
      setKeyId(patch.key ?? null);
    }
    replaceWindowSearchParams((params) => applyUsagePatchToParams(params, patch));
  }, []);

  return { view, teamId, keyId, updateSearchParams };
}
