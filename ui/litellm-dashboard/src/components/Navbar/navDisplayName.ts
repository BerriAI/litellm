import { englishTranslate, type Translate } from "@/i18n/language";

/** 导航账户名称仅翻译本地占位符，不翻译邮箱或真实用户 ID。 */
export function navAccountDisplayName(
  userEmail: string | null,
  userId: string | null,
  t: Translate = englishTranslate,
): string {
  const email = userEmail?.trim();
  if (email) {
    return email;
  }
  const id = userId?.trim();
  if (!id) {
    return t("Account");
  }
  if (/^default[_\s-]?user[_\s-]?id$/i.test(id)) {
    return t("Account");
  }
  return id;
}
