import { Badge } from "antd";
import { useDisableShowNewBadge } from "@/app/(dashboard)/hooks/useDisableShowNewBadge";

export default function NewBadge({ children }: { children?: React.ReactNode }) {
  const disableShowNewBadge = useDisableShowNewBadge();

  if (disableShowNewBadge) {
    return children ? <>{children}</> : null;
  }

  return children ? (
    <Badge color="blue" count="New">
      {children}
    </Badge>
  ) : (
    <Badge color="blue" count="New" />
  );
}
