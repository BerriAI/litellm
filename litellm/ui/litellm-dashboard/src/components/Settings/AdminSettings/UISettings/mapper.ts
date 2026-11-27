export const toPageVisibilityFields = (enabledPages: string[] | null | undefined) => ({
  selectedPages: enabledPages ?? [],
});

export const toPageVisibilityUpdate = ({ selectedPages }: ReturnType<typeof toPageVisibilityFields>) => ({
  enabled_ui_pages_internal_users: selectedPages.length > 0 ? selectedPages : null,
});
