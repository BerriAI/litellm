import { Badge } from "@/components/ui/badge";
import { useDisableShowNewBadge } from "@/app/(dashboard)/hooks/useDisableShowNewBadge";
import { useTranslation } from "react-i18next";

export default function NewBadge({ children, dot = false }: { children?: React.ReactNode; dot?: boolean }) {
  const disableShowNewBadge = useDisableShowNewBadge();
  const { t } = useTranslation();

  if (disableShowNewBadge) {
    return children ? <>{children}</> : null;
  }

  const badge = dot ? <Badge className="size-1.5 p-0" /> : <Badge>{t("common.new")}</Badge>;

  return children ? (
    <span className="relative inline-flex">
      {children}
      <span className="absolute -top-0.5 -right-1">{badge}</span>
    </span>
  ) : (
    badge
  );
}
