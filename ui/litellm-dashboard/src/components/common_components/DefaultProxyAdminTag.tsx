import { useLanguage } from "@/contexts/LanguageContext";
import { Badge } from "@/components/ui/badge";
import { DEFAULT_PROXY_ADMIN_USER_ID } from "@/utils/sentinels";

interface DefaultProxyAdminTagProps {
  userId: string | null | undefined;
}

export default function DefaultProxyAdminTag({ userId }: DefaultProxyAdminTagProps) {
  const { t } = useLanguage();
  if (userId === DEFAULT_PROXY_ADMIN_USER_ID) {
    return <Badge variant="secondary">{t("Default Proxy Admin")}</Badge>;
  }

  return <span>{userId}</span>;
}
