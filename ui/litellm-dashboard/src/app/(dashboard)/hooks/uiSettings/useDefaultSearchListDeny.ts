import { useUISettings } from "./useUISettings";

export const DEFAULT_SEARCH_LIST_DENY_SETTING_KEY = "default_search_list_deny";

export const useDefaultSearchListDeny = (): boolean =>
  useUISettings().data?.values?.[DEFAULT_SEARCH_LIST_DENY_SETTING_KEY] === true;
