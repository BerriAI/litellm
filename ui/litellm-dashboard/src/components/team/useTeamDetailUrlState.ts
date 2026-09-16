import { parseAsString, useQueryState, useQueryStates } from "nuqs";
import { useCallback, useEffect, useMemo } from "react";

import type { UrlTableStateOptions } from "@/components/shared/DataTable";
import { useUrlTab } from "@/hooks/useUrlTab";
import { useVisitedTabs } from "@/hooks/useVisitedTabs";

import {
  getTeamInfoDefaultTab,
  getTeamInfoVisibleTabs,
  TEAM_INFO_TAB_KEYS,
  type TeamInfoTabKey,
} from "./tabVisibilityUtils";

const TEAM_TAB_URL_KEY = "team_tab";
const SELECTED_KEY_URL_KEY = "key";
const KEY_DETAIL_URL_KEYS = ["key_tab", "key_savings_view"] as const;

const TEAM_KEYS_FILTER_COLUMNS = ["user_id", "key_hash"] as const;
export type TeamKeysFilterColumn = (typeof TEAM_KEYS_FILTER_COLUMNS)[number];
type TeamKeysStateKey = keyof NonNullable<UrlTableStateOptions<TeamKeysFilterColumn>["urlKeys"]>;

const TEAM_KEYS_URL_KEYS: Record<TeamKeysStateKey, string> = {
  search: "keys_search",
  sort_by: "keys_sort_by",
  sort_order: "keys_sort_order",
  page: "keys_page",
  page_size: "keys_page_size",
  filter_user_id: "keys_filter_user",
  filter_key_hash: "keys_filter_key_id",
};

export const TEAM_KEYS_TABLE_STATE_OPTIONS: UrlTableStateOptions<TeamKeysFilterColumn> = {
  sortFields: ["token", "key_alias", "created_at", "updated_at", "spend", "max_budget"],
  defaultSort: { id: "created_at", desc: true },
  defaultPageSize: 50,
  maxPageSize: 100,
  filterColumns: TEAM_KEYS_FILTER_COLUMNS,
  urlKeys: TEAM_KEYS_URL_KEYS,
};

const ALL_TEAM_INFO_TABS = getTeamInfoVisibleTabs(true);

const TEAM_DETAIL_PARSERS = Object.fromEntries(
  [TEAM_TAB_URL_KEY, SELECTED_KEY_URL_KEY, ...KEY_DETAIL_URL_KEYS, ...Object.values(TEAM_KEYS_URL_KEYS)].map((key) => [
    key,
    parseAsString,
  ]),
);

const TEAM_SELECTION_PARSERS = {
  team: parseAsString.withOptions({ history: "push" }),
  [TEAM_TAB_URL_KEY]: parseAsString,
};

const SELECTED_KEY_PARSERS = {
  [SELECTED_KEY_URL_KEY]: parseAsString.withOptions({ history: "push" }),
  [TEAM_TAB_URL_KEY]: parseAsString,
};

interface TeamDetailUrlStateOptions {
  canEditTeam: boolean;
  editTeam: boolean;
  permissionsResolved: boolean;
  onClose: () => void;
}

export interface TeamDetailUrlState {
  visibleTabs: readonly TeamInfoTabKey[];
  tab: TeamInfoTabKey;
  selectTab: (value: unknown) => void;
  hasVisited: (tab: string) => boolean;
  close: () => void;
}

export function useTeamDetailUrlState({
  canEditTeam,
  editTeam,
  permissionsResolved,
  onClose,
}: TeamDetailUrlStateOptions): TeamDetailUrlState {
  const visibleTabs = useMemo(() => getTeamInfoVisibleTabs(canEditTeam), [canEditTeam]);
  const [selectedKeyId] = useQueryState(SELECTED_KEY_URL_KEY);
  const [tab, setTab] = useUrlTab(
    permissionsResolved ? visibleTabs : ALL_TEAM_INFO_TABS,
    selectedKeyId ? TEAM_INFO_TAB_KEYS.VIRTUAL_KEYS : getTeamInfoDefaultTab(editTeam, canEditTeam),
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
  const close = useCallback(() => {
    void setDetailState(null);
    onClose();
  }, [setDetailState, onClose]);

  return { visibleTabs, tab, selectTab, hasVisited, close };
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

export interface SelectedTeamKey {
  keyId: string | null;
  openKey: (keyId: string) => void;
  closeKey: () => void;
  replaceKey: (keyId: string) => void;
}

export function useSelectedTeamKey(): SelectedTeamKey {
  const [{ key }, setSelection] = useQueryStates(SELECTED_KEY_PARSERS);

  const openKey = useCallback(
    (keyId: string) => {
      void setSelection({ [SELECTED_KEY_URL_KEY]: keyId });
    },
    [setSelection],
  );

  const closeKey = useCallback(() => {
    void setSelection({ [SELECTED_KEY_URL_KEY]: null, [TEAM_TAB_URL_KEY]: TEAM_INFO_TAB_KEYS.VIRTUAL_KEYS });
  }, [setSelection]);

  const replaceKey = useCallback(
    (keyId: string) => {
      void setSelection({ [SELECTED_KEY_URL_KEY]: keyId }, { history: "replace" });
    },
    [setSelection],
  );

  return { keyId: key, openKey, closeKey, replaceKey };
}
