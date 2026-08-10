import { Badge } from "antd";
import { useDisableShowNewBadge } from "@/app/(dashboard)/hooks/useDisableShowNewBadge";
import { useTranslation } from "react-i18next";

export default function BetaBadge({
  children,
  dot = false,
  label,
}: {
  children?: React.ReactNode;
  dot?: boolean;
  label?: React.ReactNode;
}) {
  const disableShowNewBadge = useDisableShowNewBadge();
  const { t } = useTranslation("common");
  const badgeLabel = label ?? t("badges.beta");

  if (disableShowNewBadge) {
    return children ? <>{children}</> : null;
  }

  return children ? (
    <Badge color="blue" count={dot ? undefined : badgeLabel} dot={dot}>
      {children}
    </Badge>
  ) : (
    <Badge color="blue" count={dot ? undefined : badgeLabel} dot={dot} />
  );
}
