import { parseAsBoolean, parseAsString, parseAsStringLiteral, useQueryState } from "nuqs";
import { useCallback } from "react";

import { useUrlTab } from "@/hooks/useUrlTab";
import { rolesWithWriteAccess } from "@/utils/roles";

export const USER_DETAIL_TABS = ["overview", "details"] as const;
export type UserDetailTab = (typeof USER_DETAIL_TABS)[number];

export const USER_DETAIL_URL_PARSERS = {
  user: parseAsString.withOptions({ history: "push" }),
  user_tab: parseAsStringLiteral(USER_DETAIL_TABS),
  edit: parseAsBoolean,
};

interface UserDetailUrlStateOptions {
  userRole: string | null;
  defaultTab: UserDetailTab;
  defaultEditing: boolean;
}

export interface UserDetailUrlState {
  tab: UserDetailTab;
  setTab: (tab: UserDetailTab) => void;
  isEditing: boolean;
  setIsEditing: (editing: boolean) => void;
}

export function useUserDetailUrlState({
  userRole,
  defaultTab,
  defaultEditing,
}: UserDetailUrlStateOptions): UserDetailUrlState {
  const [tab, setTab] = useUrlTab(USER_DETAIL_TABS, defaultTab, "user_tab");
  const [editRequested, setEditRequested] = useQueryState("edit", parseAsBoolean.withDefault(defaultEditing));
  const canEdit = userRole !== null && rolesWithWriteAccess.includes(userRole);
  const setIsEditing = useCallback((editing: boolean) => void setEditRequested(editing), [setEditRequested]);
  return { tab, setTab, isEditing: canEdit && editRequested, setIsEditing };
}
