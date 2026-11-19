import { useTranslation } from "react-i18next";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import type { components } from "@/lib/http/schema";

export type CustomerRateLimits = Pick<components["schemas"]["Budget"], "tpm_limit" | "rpm_limit">;

const rateFields = ["tpm", "rpm"] as const;

const hasCustomerRateLimits = (limits?: CustomerRateLimits | null): boolean =>
  [limits?.tpm_limit, limits?.rpm_limit].some((limit) => limit != null);

export function CustomerRateLimitFields({
  id,
  tpm,
  rpm,
  onChange,
  label,
  showHint = true,
}: {
  id: string;
  tpm: string;
  rpm: string;
  onChange: (field: "tpm" | "rpm", value: string) => void;
  label?: string;
  showHint?: boolean;
}) {
  const { t } = useTranslation();
  const values = { tpm, rpm };
  return (
    <fieldset className="space-y-2">
      <legend className="text-sm font-medium">{label ?? t("customerPortal.rateLimits.title")}</legend>
      <div className="grid gap-3 sm:grid-cols-2">
        {rateFields.map((field) => (
          <div key={field} className="space-y-1.5">
            <Label htmlFor={`${id}-${field}`}>{t(`customerPortal.hierarchy.${field}`)}</Label>
            <Input
              id={`${id}-${field}`}
              type="number"
              min={0}
              step={1}
              value={values[field]}
              placeholder={t("customerPortal.createDialog.noLimit")}
              onChange={(event) => onChange(field, event.target.value)}
            />
          </div>
        ))}
      </div>
      {showHint && <p className="text-xs text-muted-foreground">{t("customerPortal.rateLimits.hint")}</p>}
    </fieldset>
  );
}

export function CustomerRateLimitSummary({ limits }: { limits?: CustomerRateLimits | null }) {
  const { t } = useTranslation();
  if (!hasCustomerRateLimits(limits)) return null;
  return (
    <p className="flex flex-wrap gap-x-3 text-xs text-muted-foreground tabular-nums">
      {limits?.tpm_limit != null && (
        <span>{t("customerPortal.detail.tpm", { val: t("format.number", { val: limits.tpm_limit }) })}</span>
      )}
      {limits?.rpm_limit != null && (
        <span>{t("customerPortal.detail.rpm", { val: t("format.number", { val: limits.rpm_limit }) })}</span>
      )}
    </p>
  );
}
