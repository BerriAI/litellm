export interface SearchToolPermissionCopy {
  hint: string;
  placeholder: string;
  emptyState: string;
}

export const searchToolPermissionCopy = (defaultSearchListDeny: boolean): SearchToolPermissionCopy =>
  defaultSearchListDeny
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
