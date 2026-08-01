import { Badge } from "antd";
import { useDisableShowNewBadge } from "@/app/(dashboard)/hooks/useDisableShowNewBadge";

export default function BetaBadge({ children, dot = false }: { children?: React.ReactNode; dot?: boolean }) {
  const disableShowNewBadge = useDisableShowNewBadge();

  if (disableShowNewBadge) {
    return children ? <>{children}</> : null;
  }

  return children ? (
    <Badge color="blue" count={dot ? undefined : "Beta"} dot={dot}>
      {children}
    </Badge>
  ) : (
    <Badge color="blue" count={dot ? undefined : "Beta"} dot={dot} />
  );
}
