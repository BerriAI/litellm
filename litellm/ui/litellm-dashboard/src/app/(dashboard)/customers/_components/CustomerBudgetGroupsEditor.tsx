import { Plus, Trash2 } from "lucide-react";
import { useTranslation } from "react-i18next";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { InputGroup, InputGroupAddon, InputGroupInput, InputGroupText } from "@/components/ui/input-group";
import { Label } from "@/components/ui/label";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { defaultCustomerCapPolicy, type CustomerBudgetGroupFields } from "./customerOrganizationForm";

export function BudgetInput({
  id,
  label,
  value,
  onChange,
}: {
  id: string;
  label: string;
  value: string;
  onChange: (value: string) => void;
}) {
  const { t } = useTranslation();
  return (
    <InputGroup>
      <InputGroupAddon>
        <InputGroupText>$</InputGroupText>
      </InputGroupAddon>
      <InputGroupInput
        id={id}
        aria-label={label}
        type="number"
        min={0}
        step="any"
        value={value}
        onChange={(event) => onChange(event.target.value)}
        placeholder={t("customerPortal.createDialog.noLimit")}
      />
    </InputGroup>
  );
}

const groupBudgetFields = [
  ["monthlyBudget", "monthly"],
  ["yearlyBudget", "yearly"],
] as const;

export default function CustomerBudgetGroupsEditor({
  groups,
  defaultGroupId,
  onChange,
  disabled = false,
}: {
  groups: CustomerBudgetGroupFields[];
  defaultGroupId: string;
  onChange: (groups: CustomerBudgetGroupFields[], defaultGroupId: string) => void;
  disabled?: boolean;
}) {
  const { t } = useTranslation();
  const updateGroup = (id: string, patch: Partial<Omit<CustomerBudgetGroupFields, "id">>) =>
    onChange(
      groups.map((group) => (group.id === id ? { ...group, ...patch } : group)),
      defaultGroupId,
    );
  const removeGroup = (id: string) => {
    const remaining = groups.filter((group) => group.id !== id);
    onChange(remaining, defaultGroupId === id ? remaining[0]?.id ?? "" : defaultGroupId);
  };
  const addGroup = () => {
    const group: CustomerBudgetGroupFields = {
      id: crypto.randomUUID(),
      name: "",
      monthlyBudget: "",
      yearlyBudget: "",
      capPolicy: defaultCustomerCapPolicy,
    };
    onChange([...groups, group], groups.length ? defaultGroupId : group.id);
  };
  return (
    <section className="space-y-3 border-t pt-5" aria-labelledby="cc-groups-title">
      <div>
        <h3 id="cc-groups-title" className="font-medium">
          {t("customerPortal.createDialog.groups")}
        </h3>
        <p className="mt-1 text-xs text-muted-foreground">{t("customerPortal.createDialog.groupsHelp")}</p>
      </div>
      <div
        className="hidden grid-cols-[1.3fr_1fr_1fr_3rem_2rem] gap-3 px-2 text-xs text-muted-foreground sm:grid"
        aria-hidden="true"
      >
        <span>{t("customerPortal.createDialog.groupName")}</span>
        <span>{t("customerPortal.createDialog.monthly")}</span>
        <span>{t("customerPortal.createDialog.yearly")}</span>
        <span>{t("customerPortal.createDialog.default")}</span>
      </div>
      <RadioGroup
        value={defaultGroupId}
        onValueChange={(value) => onChange(groups, String(value))}
        aria-label={t("customerPortal.createDialog.defaultGroup")}
        disabled={disabled}
        className="gap-1"
      >
        {groups.map((group, index) => (
          <div
            key={group.id}
            className={`grid grid-cols-2 items-center gap-3 rounded-md p-2 sm:grid-cols-[1.3fr_1fr_1fr_3rem_2rem] ${group.id === defaultGroupId ? "bg-muted/60" : ""}`}
          >
            <div className="col-span-2 min-w-0 space-y-1.5 sm:col-span-1">
              <Label htmlFor={`cc-group-${group.id}`} className="sm:sr-only">
                {t("customerPortal.createDialog.groupNameLabel", { index: index + 1 })}
              </Label>
              <Input
                id={`cc-group-${group.id}`}
                value={group.name}
                onChange={(event) => updateGroup(group.id, { name: event.target.value })}
                className="font-mono text-xs"
              />
            </div>
            {groupBudgetFields.map(([field, period]) => (
              <div className="min-w-0 space-y-1.5" key={field}>
                <Label htmlFor={`cc-${period}-${group.id}`} className="sm:sr-only">
                  {t(`customerPortal.createDialog.${period}`)}
                </Label>
                <BudgetInput
                  id={`cc-${period}-${group.id}`}
                  label={t(`customerPortal.createDialog.${period}Label`, { index: index + 1 })}
                  value={group[field]}
                  onChange={(value) => updateGroup(group.id, { [field]: value })}
                />
              </div>
            ))}
            <div className="col-span-2 min-w-0 space-y-1.5 sm:col-span-3 sm:row-start-2">
              <Label htmlFor={`cc-cap-${group.id}`}>{t("customerPortal.budgetGroups.atCap")}</Label>
              <Select
                value={group.capPolicy}
                disabled={disabled}
                onValueChange={(value) => {
                  if (value === "pause_until_reset" || value === "pause_allow_increase") {
                    updateGroup(group.id, { capPolicy: value });
                  }
                }}
              >
                <SelectTrigger
                  id={`cc-cap-${group.id}`}
                  aria-label={t("customerPortal.budgetGroups.atCapLabel", { index: index + 1 })}
                  className="w-full min-w-0 text-xs"
                >
                  <SelectValue>
                    {t(
                      group.capPolicy === "pause_allow_increase"
                        ? "customerPortal.budgetGroups.pauseAllowIncrease"
                        : "customerPortal.budgetGroups.pauseUntilReset",
                    )}
                  </SelectValue>
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="pause_until_reset">{t("customerPortal.budgetGroups.pauseUntilReset")}</SelectItem>
                  <SelectItem value="pause_allow_increase">
                    {t("customerPortal.budgetGroups.pauseAllowIncrease")}
                  </SelectItem>
                </SelectContent>
              </Select>
            </div>
            <Label className="flex gap-2 sm:col-start-4 sm:row-start-1 sm:justify-center">
              <RadioGroupItem
                value={group.id}
                aria-label={t("customerPortal.createDialog.useDefault", { name: group.name || index + 1 })}
              />
              <span className="text-xs sm:sr-only" aria-hidden="true">
                {t("customerPortal.createDialog.default")}
              </span>
            </Label>
            <Button
              type="button"
              variant="ghost"
              size="icon-sm"
              className="justify-self-end sm:col-start-5 sm:row-start-1"
              disabled={groups.length === 1}
              onClick={() => removeGroup(group.id)}
              aria-label={t("customerPortal.createDialog.removeGroup", { name: group.name || index + 1 })}
            >
              <Trash2 className="size-3.5" />
            </Button>
          </div>
        ))}
      </RadioGroup>
      <Button type="button" variant="ghost" size="sm" onClick={addGroup}>
        <Plus className="size-4" />
        {t("customerPortal.createDialog.addGroup")}
      </Button>
      <p className="text-xs text-muted-foreground">{t("customerPortal.createDialog.defaultHelp")}</p>
      <p className="text-xs text-muted-foreground">{t("customerPortal.createDialog.resetHelp")}</p>
    </section>
  );
}
