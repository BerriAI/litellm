import { Badge } from "antd";
import { useDisableShowBadges } from "@/app/(dashboard)/hooks/useDisableShowBadges";

export default function NewBadge({ children, dot = false }: { children?: React.ReactNode; dot?: boolean }) {
  const disableShowBadges = useDisableShowBadges();

  if (disableShowBadges) {
    return children ? <>{children}</> : null;
  }

  return children ? (
    <Badge color="blue" count={dot ? undefined : "New"} dot={dot}>
      {children}
    </Badge>
  ) : (
    <Badge color="blue" count={dot ? undefined : "New"} dot={dot} />
  );
}
