import type { ComponentProps } from "react";
import { useTranslation } from "react-i18next";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import type { components } from "@/lib/http/schema";

export type BudgetCapPolicy = NonNullable<components["schemas"]["Budget"]["cap_policy"]>;

type Props = Omit<ComponentProps<typeof SelectTrigger>, "value" | "onChange" | "children"> & {
  value: BudgetCapPolicy;
  onValueChange: (value: BudgetCapPolicy) => void;
};

export function BudgetCapPolicySelect({ value, onValueChange, disabled, ...triggerProps }: Props) {
  const { t } = useTranslation();
  const items = [
    { value: "pause_allow_increase", label: t("budgetProfiles.contactAdmin") },
    { value: "pause_until_reset", label: t("budgetProfiles.wait") },
  ];
  return (
    <Select
      value={value}
      onValueChange={(next) => {
        if (next === "pause_allow_increase" || next === "pause_until_reset") onValueChange(next);
      }}
      disabled={disabled}
      items={items}
    >
      <SelectTrigger className="h-auto min-h-9 w-full whitespace-normal" {...triggerProps}>
        <SelectValue />
      </SelectTrigger>
      <SelectContent>
        {items.map((item) => (
          <SelectItem key={item.value} value={item.value}>
            <span className="whitespace-normal">{item.label}</span>
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}
