import { Badge } from "@/components/ui/badge";

const DEFAULT_USER_ID = "default_user_id";

interface DefaultProxyAdminTagProps {
  userId: string | null | undefined;
}

export default function DefaultProxyAdminTag({ userId }: DefaultProxyAdminTagProps) {
  if (userId === DEFAULT_USER_ID) {
    return <Badge variant="secondary">Default Proxy Admin</Badge>;
  }

  return <span>{userId}</span>;
}
