import { Badge } from "@/components/ui/badge";

const DEFAULT_USER_ID = "default_user_id";

interface DefaultProxyAdminTagProps {
  userId: string | null | undefined;
}

/**
 * Renders "Default Proxy Admin" as a primary-toned Badge when the given
 * userId is the well-known `default_user_id`, otherwise renders the raw
 * value as plain text.
 */
export default function DefaultProxyAdminTag({
  userId,
}: DefaultProxyAdminTagProps) {
  if (userId === DEFAULT_USER_ID) {
    return <Badge variant="default">Default Proxy Admin</Badge>;
  }

  return <span>{userId}</span>;
}
