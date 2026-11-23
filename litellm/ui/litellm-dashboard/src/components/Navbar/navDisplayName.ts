import i18n from "@/i18n/i18n";

/** Primary label for the navbar account control — avoids raw placeholder JWT/user IDs in the UI. */
export function navAccountDisplayName(
  userEmail: string | null,
  userId: string | null,
  userAlias?: string | null,
): string {
  const identity = userEmail?.trim() || userAlias?.trim();
  if (identity) {
    return identity;
  }
  const id = userId?.trim();
  if (!id) {
    return i18n.t("shell.navDisplayName.account");
  }
  if (/^default[_\s-]?user[_\s-]?id$/i.test(id)) {
    return i18n.t("shell.navDisplayName.account");
  }
  return id;
}
