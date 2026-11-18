import { useTranslation } from "react-i18next";
import { $api } from "@/lib/http/api";
import { SearchSelect } from "@/components/shared/SearchSelect";
import { ErrorState } from "@/components/shared/ErrorState";
import { Label } from "@/components/ui/label";

export default function CustomerBudgetInput({
  id,
  budgetId,
  amount,
  onChange,
}: {
  id: string;
  budgetId: string;
  amount: string;
  onChange: (budgetId: string, amount: string) => void;
}) {
  const { t } = useTranslation();
  const groups = $api.useQuery("get", "/management/v1/customer_organization_budget_groups", {});
  const options = (groups.data?.default_budgets ?? []).map((group) => ({
    value: group.budget_id,
    label: group.name,
  }));
  const selectedGroup = options.find((group) => group.value === budgetId);
  const selectedAmount = Number.isFinite(Number(amount)) ? amount || null : null;
  const customAmount = (text: string) => onChange(selectedGroup ? "" : budgetId, text.replace(/^\$\s*/, ""));
  return (
    <div className="space-y-1.5">
      <Label htmlFor={id}>{t("customerPortal.createDialog.maxBudget")}</Label>
      <SearchSelect
        id={id}
        options={options}
        value={selectedGroup ? budgetId : selectedAmount}
        placeholder={t("customerPortal.organizationGroups.placeholder")}
        disabled={!groups.data}
        inputValue={selectedGroup?.label ?? amount}
        onInputChange={customAmount}
        customValue={{
          hint: t("customerPortal.organizationGroups.custom"),
          accepts: (value) =>
            /^\$?(?:\d+(?:\.\d*)?|\.\d+)$/.test(value) && Number.isFinite(Number(value.replace("$", ""))),
        }}
        onValueChange={(value) =>
          options.some((group) => group.value === value) ? onChange(value ?? "", amount) : customAmount(value ?? "")
        }
      />
      {groups.isError && <ErrorState onRetry={() => void groups.refetch()} />}
    </div>
  );
}
