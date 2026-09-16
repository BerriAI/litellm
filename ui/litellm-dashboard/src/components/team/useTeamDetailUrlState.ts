import { parseAsString, useQueryStates } from "nuqs";
import { useCallback, useEffect } from "react";

import { useUrlTab } from "@/hooks/useUrlTab";
import { useVisitedTabs } from "@/hooks/useVisitedTabs";

import { getTeamInfoVisibleTabs, TEAM_INFO_TAB_KEYS, type TeamInfoTabKey } from "./tabVisibilityUtils";

const TEAM_TAB_URL_KEY = "team_tab";
export const SELECTED_TEAM_KEY_URL_KEY = "key";
export const TEAM_KEYS_URL_PREFIX = "keys_";
export const TEAM_KEYS_FILTER_URL_KEYS = { filter_user_id: "filter_user", filter_key_hash: "filter_key_id" } as const;

const TEAM_KEYS_TABLE_URL_KEYS = [
  "search",
  "sort_by",
  "sort_order",
  "page",
  "page_size",
  ...Object.values(TEAM_KEYS_FILTER_URL_KEYS),
].map((key) => `${TEAM_KEYS_URL_PREFIX}${key}`);

const ALL_TEAM_INFO_TABS = getTeamInfoVisibleTabs(true);

const TEAM_DETAIL_PARSERS = Object.fromEntries(
  [TEAM_TAB_URL_KEY, SELECTED_TEAM_KEY_URL_KEY, ...TEAM_KEYS_TABLE_URL_KEYS].map((key) => [key, parseAsString]),
);

const TEAM_SELECTION_PARSERS = {
  team: parseAsString.withOptions({ history: "push" }),
  [TEAM_TAB_URL_KEY]: parseAsString,
};

export interface TeamDetailUrlState {
  tab: TeamInfoTabKey;
  selectTab: (value: unknown) => void;
  hasVisited: (tab: string) => boolean;
  clearDetailState: () => void;
}

export function useTeamDetailUrlState(
  visibleTabs: readonly TeamInfoTabKey[],
  fallbackTab: TeamInfoTabKey,
  permissionsResolved: boolean,
): TeamDetailUrlState {
  const [tab, setTab] = useUrlTab(
    permissionsResolved ? visibleTabs : ALL_TEAM_INFO_TABS,
    fallbackTab,
    TEAM_TAB_URL_KEY,
  );
  const { onTabChange, hasVisited } = useVisitedTabs(tab);
  useEffect(() => {
    onTabChange(tab);
  }, [tab, onTabChange]);

  const selectTab = useCallback(
    (value: unknown) => {
      const next = ALL_TEAM_INFO_TABS.find((key) => key === value);
      if (next) setTab(next);
    },
    [setTab],
  );

  const [, setDetailState] = useQueryStates(TEAM_DETAIL_PARSERS);
  const clearDetailState = useCallback(() => {
    void setDetailState(null);
  }, [setDetailState]);

  return { tab, selectTab, hasVisited, clearDetailState };
}

export interface TeamSelection {
  teamId: string | null;
  openTeam: (teamId: string) => void;
  editTeam: (teamId: string) => void;
  closeTeam: () => void;
}

export function useTeamSelection(): TeamSelection {
  const [{ team }, setSelection] = useQueryStates(TEAM_SELECTION_PARSERS);

  const openTeam = useCallback(
    (teamId: string) => {
      void setSelection({ team: teamId, [TEAM_TAB_URL_KEY]: null });
    },
    [setSelection],
  );

  const editTeam = useCallback(
    (teamId: string) => {
      void setSelection({ team: teamId, [TEAM_TAB_URL_KEY]: TEAM_INFO_TAB_KEYS.SETTINGS });
    },
    [setSelection],
  );

  const closeTeam = useCallback(() => {
    void setSelection(null);
  }, [setSelection]);

  return { teamId: team, openTeam, editTeam, closeTeam };
}
