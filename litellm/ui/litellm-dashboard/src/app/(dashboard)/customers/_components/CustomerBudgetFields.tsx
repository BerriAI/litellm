import { useTranslation } from "react-i18next";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import type { BudgetFields } from "./customerBudgetForm";
import CustomerAllowedModelsSelect from "./CustomerAllowedModelsSelect";
import CustomerBudgetAmountInput from "./CustomerBudgetAmountInput";
import CustomerBudgetPeriodSelect from "./CustomerBudgetPeriodSelect";
import { CustomerRateLimitControls, CustomerRateLimitFields } from "./CustomerRateLimits";

interface Props {
  fields: BudgetFields;
  onChange: {
    (key: Exclude<keyof BudgetFields, "allowedModels">, value: string): void;
    (key: "allowedModels", value: readonly string[]): void;
  };
  idPrefix: string;
  disabled?: boolean;
}

function AmountField({ fields, onChange, idPrefix, disabled, name }: Props & { name: "maxBudget" | "softBudget" }) {
  const { t } = useTranslation();
  const label = t(`customerPortal.hierarchy.${name}`);
  return (
    <div className="space-y-2">
      <Label htmlFor={`${idPrefix}-${name}`} className="font-normal text-muted-foreground">
        {label}
      </Label>
      <CustomerBudgetAmountInput
        id={`${idPrefix}-${name}`}
        label={label}
        disabled={disabled}
        value={fields[name]}
        onChange={(value) => onChange(name, value)}
      />
    </div>
  );
}

function ModelsField({ fields, onChange, idPrefix, disabled }: Props) {
  const { t } = useTranslation();
  return (
    <div className="space-y-2">
      <Label htmlFor={`${idPrefix}-models`} className="font-normal text-muted-foreground">
        {t("customerPortal.hierarchy.allowedModels")}
      </Label>
      <CustomerAllowedModelsSelect
        id={`${idPrefix}-models`}
        disabled={disabled}
        value={fields.allowedModels}
        onChange={(models) => onChange("allowedModels", models)}
      />
      <p className="text-xs text-muted-foreground">{t("customerPortal.hierarchy.allowedModelsHint")}</p>
    </div>
  );
}

export function CustomerBudgetAdvancedFields(props: Props) {
  return (
    <div className="grid gap-4 sm:grid-cols-2">
      <AmountField {...props} name="softBudget" />
      <ModelsField {...props} />
    </div>
  );
}

export function CustomerBudgetLimitFields({
  fields,
  onChange,
  idPrefix,
  disabled = false,
  period = "custom",
}: {
  fields: Pick<BudgetFields, "maxBudget" | "budgetDuration" | "tpm" | "rpm">;
  onChange: (key: "maxBudget" | "budgetDuration" | "tpm" | "rpm", value: string) => void;
  idPrefix: string;
  disabled?: boolean;
  period?: "standard" | "custom";
}) {
  const { t } = useTranslation();
  return (
    <div className="grid gap-4 sm:grid-cols-[1.5fr_1fr_1fr]">
      <div className="min-w-0 space-y-2">
        <Label htmlFor={`${idPrefix}-budget`} className="font-normal text-muted-foreground">
          {t("customerPortal.createDialog.budget")}
        </Label>
        <CustomerBudgetAmountInput
          id={`${idPrefix}-budget`}
          label={t("customerPortal.createDialog.spendLimitLabel")}
          disabled={disabled}
          value={fields.maxBudget}
          onChange={(value) => onChange("maxBudget", value)}
        >
          <CustomerBudgetPeriodSelect
            value={fields.budgetDuration}
            disabled={disabled}
            standardOnly={period === "standard"}
            onChange={(value) => onChange("budgetDuration", value)}
          />
        </CustomerBudgetAmountInput>
      </div>
      <CustomerRateLimitControls
        id={idPrefix}
        tpm={fields.tpm}
        rpm={fields.rpm}
        disabled={disabled}
        onChange={onChange}
      />
    </div>
  );
}

export function CustomerBudgetFields({ layout = "detailed", ...props }: Props & { layout?: "scalar" | "detailed" }) {
  const { t } = useTranslation();
  const { fields, onChange, idPrefix, disabled = false } = props;
  if (layout === "scalar")
    return (
      <div className="space-y-5">
        <CustomerBudgetLimitFields {...props} />
        <div className="border-t pt-5">
          <CustomerBudgetAdvancedFields {...props} />
        </div>
      </div>
    );
  return (
    <div className="space-y-4">
      <div className="grid gap-4 sm:grid-cols-2">
        <AmountField {...props} name="maxBudget" />
        <AmountField {...props} name="softBudget" />
        <div className="space-y-1.5">
          <Label htmlFor={`${idPrefix}-duration`}>{t("customerPortal.hierarchy.budgetDuration")}</Label>
          <Input
            id={`${idPrefix}-duration`}
            disabled={disabled}
            value={fields.budgetDuration}
            onChange={(event) => onChange("budgetDuration", event.target.value)}
            aria-describedby={`${idPrefix}-duration-hint`}
          />
          <p id={`${idPrefix}-duration-hint`} className="text-xs text-muted-foreground">
            {t("customerPortal.hierarchy.budgetDurationHint")}
          </p>
        </div>
      </div>
      <CustomerRateLimitFields
        id={idPrefix}
        tpm={fields.tpm}
        rpm={fields.rpm}
        disabled={disabled}
        onChange={onChange}
      />
      <ModelsField {...props} />
    </div>
  );
}
