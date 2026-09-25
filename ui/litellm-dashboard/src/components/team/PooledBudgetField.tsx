import { FieldDescription, FieldLabel } from "@/components/ui/field";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { forwardRef, type ComponentProps } from "react";
import type { PooledBudgetValue } from "./pooledBudget";

interface PooledBudgetFieldProps extends Pick<ComponentProps<typeof Input>, "aria-invalid" | "aria-describedby"> {
  id: string;
  name: string;
  value: PooledBudgetValue;
  onChange: (value: PooledBudgetValue) => void;
  onBlur: () => void;
  mode: "create" | "edit";
  defaultBudget?: number | null;
  defaultBudgetStatus?: "pending" | "error" | "success";
}

const defaultBudgetDescription = (
  status: PooledBudgetFieldProps["defaultBudgetStatus"],
  budget: PooledBudgetFieldProps["defaultBudget"],
): string => {
  if (status === "pending") return "Loading the deployment's default pooled budget";
  if (status === "error") return "Deployment defaults could not be loaded. Leaving this off uses those defaults";
  if (budget != null) {
    return `Leaving this off uses the deployment's default pooled budget of $${budget.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 20 })}`;
  }
  return "Leaving this off uses the deployment's default pooled budget, if configured";
};

export const PooledBudgetField = forwardRef<HTMLInputElement, PooledBudgetFieldProps>(
  ({ id, name, value, onChange, onBlur, mode, defaultBudget, defaultBudgetStatus, ...aria }, ref) => {
    const enabled = value != null;
    const amountId = `${id}-amount`;
    const descriptionId = `${id}-pool-description`;
    const describedBy = [aria["aria-describedby"], descriptionId].filter(Boolean).join(" ");
    const unsetValue = mode === "create" ? undefined : null;

    return (
      <div className="space-y-3">
        <div className="flex items-center gap-3">
          <Switch
            id={id}
            checked={enabled}
            onCheckedChange={(checked) => onChange(checked ? "" : unsetValue)}
            onBlur={onBlur}
            aria-describedby={descriptionId}
          />
          <FieldLabel htmlFor={id}>Pooled budget</FieldLabel>
        </div>
        <FieldDescription id={descriptionId}>
          Share one spending cap across all keys and members in this team. Each member&apos;s own limit still applies.
        </FieldDescription>
        {enabled ? (
          <div className="space-y-2">
            <FieldLabel htmlFor={amountId}>Pool amount (USD)</FieldLabel>
            <Input
              ref={ref}
              id={amountId}
              name={name}
              type="number"
              min={0}
              step="any"
              className="max-w-xs"
              value={value}
              onChange={(event) => onChange(event.target.value === "" ? "" : Number(event.target.value))}
              onBlur={onBlur}
              aria-invalid={aria["aria-invalid"]}
              aria-describedby={describedBy}
            />
            <FieldDescription>
              Requests stop when the team reaches this amount. Set 0 to stop new requests.
            </FieldDescription>
          </div>
        ) : (
          <FieldDescription>
            {mode === "create"
              ? defaultBudgetDescription(defaultBudgetStatus, defaultBudget)
              : "No shared spending cap. Member, key, and organization limits still apply."}
          </FieldDescription>
        )}
      </div>
    );
  },
);

PooledBudgetField.displayName = "PooledBudgetField";
