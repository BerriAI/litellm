import { Tag, Typography } from "antd";
import { useTranslation } from "react-i18next";

const { Text } = Typography;

const DEFAULT_USER_ID = "default_user_id";

interface DefaultProxyAdminTagProps {
  userId: string | null | undefined;
}

/**
 * Renders "Default Proxy Admin" as a blue Tag when the given userId is
 * the well-known `default_user_id`, otherwise renders the raw value as
 * plain text.
 */
export default function DefaultProxyAdminTag({ userId }: DefaultProxyAdminTagProps) {
  const { t } = useTranslation();

  if (userId === DEFAULT_USER_ID) {
    return <Tag color="blue">{t("commonComponents.defaultProxyAdminTag.label")}</Tag>;
  }

  return <Text>{userId}</Text>;
}
