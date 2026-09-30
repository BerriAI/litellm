export interface SearchToolPermissionCopy {
  hint: string;
  placeholder: string;
  emptyState: string;
}

export type SearchToolPermissionScope = "team" | "key";

const keyEmptyState = (defaultSearchListDeny: boolean): string =>
  defaultSearchListDeny
    ? "No key-level search tools. Default search list deny is on, so this key can use only the search tools granted to its team or user."
    : "No key-level restriction: this key can use any search tool its team or user allows.";

export const searchToolPermissionCopy = (
  defaultSearchListDeny: boolean,
  scope: SearchToolPermissionScope = "team",
): SearchToolPermissionCopy => {
  const copy = defaultSearchListDeny
    ? {
        hint: "Select which search tools this team can access. Default search list deny is on, so leaving this empty denies every search tool.",
        placeholder: "Select search tools (empty = none allowed)",
        emptyState: "No search tools granted. Default search list deny is on, so no search tool is allowed.",
      }
    : {
        hint: "Select which search tools this team can access. Leave empty to allow all search tools.",
        placeholder: "Select search tools (optional, empty = all allowed)",
        emptyState: "No restriction: all configured search tools are allowed for this team.",
      };
  return scope === "key" ? { ...copy, emptyState: keyEmptyState(defaultSearchListDeny) } : copy;
};
