import { useTranslation } from "react-i18next";
import { Badge } from "@/components/ui/badge";
import type { CustomerStatus } from "./customerPortalData";

const VARIANT = { active: "secondary", over_budget: "destructive", blocked: "outline" } as const;
const LABEL_KEY = {
  active: "customerPortal.status.active",
  over_budget: "customerPortal.status.overBudget",
  blocked: "customerPortal.status.blocked",
} as const;

export default function StatusBadge({ status }: { status: CustomerStatus }) {
  const { t } = useTranslation();
  return <Badge variant={VARIANT[status]}>{t(LABEL_KEY[status])}</Badge>;
}
