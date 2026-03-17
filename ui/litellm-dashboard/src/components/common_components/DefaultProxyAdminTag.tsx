import { Tag, Typography } from "antd";

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
export default function DefaultProxyAdminTag({
  userId,
}: DefaultProxyAdminTagProps) {
  if (userId === DEFAULT_USER_ID) {
    return <Tag color="blue">Default Proxy Admin</Tag>;
  }

  return <Text>{userId}</Text>;
}
