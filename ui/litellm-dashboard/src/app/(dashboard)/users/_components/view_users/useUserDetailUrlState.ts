import { parseAsBoolean, parseAsString, parseAsStringLiteral, useQueryState } from "nuqs";
import { useCallback } from "react";

import { useUrlTab } from "@/hooks/useUrlTab";

export const USER_DETAIL_TABS = ["overview", "details"] as const;
export type UserDetailTab = (typeof USER_DETAIL_TABS)[number];

const USER_TAB_KEY = "user_tab";
const EDIT_KEY = "edit";

export const USER_DETAIL_URL_PARSERS = {
  user: parseAsString.withOptions({ history: "push" }),
  [USER_TAB_KEY]: parseAsStringLiteral(USER_DETAIL_TABS),
  [EDIT_KEY]: parseAsBoolean,
};

interface UserDetailUrlStateOptions {
  defaultTab: UserDetailTab;
  defaultEditing: boolean;
}

export interface UserDetailUrlState {
  tab: UserDetailTab;
  setTab: (tab: UserDetailTab) => void;
  isEditing: boolean;
  setIsEditing: (editing: boolean) => void;
}

export function useUserDetailUrlState({ defaultTab, defaultEditing }: UserDetailUrlStateOptions): UserDetailUrlState {
  const [tab, setTab] = useUrlTab(USER_DETAIL_TABS, defaultTab, USER_TAB_KEY);
  const [isEditing, setEditing] = useQueryState(EDIT_KEY, parseAsBoolean.withDefault(defaultEditing));
  const setIsEditing = useCallback((editing: boolean) => void setEditing(editing), [setEditing]);
  return { tab, setTab, isEditing, setIsEditing };
}
