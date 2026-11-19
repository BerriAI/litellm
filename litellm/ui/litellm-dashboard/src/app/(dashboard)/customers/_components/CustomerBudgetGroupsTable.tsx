import { useId } from "react";
import { Plus, X } from "lucide-react";
import { useTranslation } from "react-i18next";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import type { CustomerBudgetGroupFields } from "./customerOrganizationForm";
import CustomerBudgetCompactFields from "./CustomerBudgetCompactFields";

const capOptions = [
  ["pause_until_reset", "pauseUntilReset"],
  ["pause_allow_increase", "pauseAllowIncrease"],
] as const;

export default function CustomerBudgetGroupsTable({
  groups,
  defaultGroupId,
  disabled,
  scope,
  title,
  description,
  onUpdate,
  onRemove,
  onAdd,
  onDefaultChange,
}: {
  groups: CustomerBudgetGroupFields[];
  defaultGroupId: string;
  disabled: boolean;
  scope: "member" | "organization" | "app" | "overall-app";
  title: string;
  description: string;
  onUpdate: (id: string, patch: Partial<Omit<CustomerBudgetGroupFields, "id">>) => void;
  onRemove: (id: string) => void;
  onAdd: () => void;
  onDefaultChange: (id: string) => void;
}) {
  const { t } = useTranslation();
  const id = useId();
  const memberGroups = scope === "member";
  const groupName = (group: CustomerBudgetGroupFields, index: number) =>
    group.name || t("customerPortal.createDialog.unnamedGroup", { index: index + 1 });
  const defaultIndex = groups.findIndex((group) => group.id === defaultGroupId);
  return (
    <section className="space-y-3" aria-labelledby={`${id}-heading`}>
      <div className="space-y-1">
        <h3 id={`${id}-heading`} className="font-semibold">
          {title}
        </h3>
        <p className="text-sm text-muted-foreground">{description}</p>
      </div>
      <div className="space-y-4">
        {groups.map((group, index) => (
          <fieldset
            key={group.id}
            aria-label={groupName(group, index)}
            disabled={disabled}
            className="min-w-0 space-y-5 rounded-lg border p-4"
          >
            <div className="flex items-end gap-3">
              <div className="min-w-0 flex-1 space-y-2">
                <Label htmlFor={`${id}-name-${group.id}`} className="font-normal text-muted-foreground">
                  {t("customerPortal.createDialog.groupColumn")}
                </Label>
                <Input
                  id={`${id}-name-${group.id}`}
                  aria-label={t("customerPortal.createDialog.groupNameLabel", { index: index + 1 })}
                  value={group.name}
                  onChange={(event) => onUpdate(group.id, { name: event.target.value })}
                  className="h-11"
                />
              </div>
              <Button
                type="button"
                variant="ghost"
                size="icon-sm"
                disabled={disabled || (memberGroups && groups.length === 1)}
                onClick={() => onRemove(group.id)}
                aria-label={t("customerPortal.createDialog.removeGroup", { name: groupName(group, index) })}
                className="mb-1.5 text-muted-foreground"
              >
                <X className="size-3.5" />
              </Button>
            </div>
            <CustomerBudgetCompactFields
              fields={group}
              id={`${id}-${group.id}`}
              disabled={disabled}
              onChange={(fields) => onUpdate(group.id, fields)}
            />
            {memberGroups && (
              <div className="max-w-sm space-y-2">
                <Label htmlFor={`${id}-cap-${group.id}`} className="font-normal text-muted-foreground">
                  {t("customerPortal.createDialog.capColumn")}
                </Label>
                <Select
                  value={group.capPolicy}
                  disabled={disabled}
                  onValueChange={(value) => {
                    if (value === "pause_until_reset" || value === "pause_allow_increase") {
                      onUpdate(group.id, { capPolicy: value });
                    }
                  }}
                >
                  <SelectTrigger
                    id={`${id}-cap-${group.id}`}
                    aria-label={t("customerPortal.budgetGroups.atCapLabel", { index: index + 1 })}
                    className="w-full"
                  >
                    <SelectValue>
                      {t(
                        group.capPolicy === "pause_allow_increase"
                          ? "customerPortal.createDialog.pauseRequest"
                          : "customerPortal.createDialog.pause",
                      )}
                    </SelectValue>
                  </SelectTrigger>
                  <SelectContent align="start" className="w-80 max-w-[calc(100vw-2rem)] p-1">
                    {capOptions.map(([policy, key]) => (
                      <SelectItem
                        key={policy}
                        value={policy}
                        label={t(`customerPortal.createDialog.${key}`)}
                        aria-label={t(`customerPortal.createDialog.${key}`)}
                        className="items-start rounded-md py-3"
                      >
                        <span className="space-y-1 whitespace-normal">
                          <span className="block font-medium">{t(`customerPortal.createDialog.${key}`)}</span>
                          <span className="block text-xs leading-relaxed text-muted-foreground">
                            {t(`customerPortal.createDialog.${key}Help`)}
                          </span>
                        </span>
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            )}
          </fieldset>
        ))}
        <Button type="button" variant="outline" size="sm" disabled={disabled} onClick={onAdd}>
          <Plus className="size-4" />
          {t("customerPortal.createDialog.addGroup")}
        </Button>
      </div>
      {(memberGroups || scope === "organization") && (
        <div className="grid items-center gap-3 rounded-lg bg-muted/40 p-4 sm:grid-cols-[1fr_14rem]">
          <div className="space-y-1">
            <Label htmlFor={`${id}-default-group`}>
              {t(
                memberGroups
                  ? "customerPortal.createDialog.newMembersJoin"
                  : "customerPortal.budgetGroups.newOrganizationsJoin",
              )}
            </Label>
            <p className="text-xs text-muted-foreground" id={`${id}-default-group-help`}>
              {t(
                memberGroups
                  ? "customerPortal.createDialog.newMembersHelp"
                  : "customerPortal.budgetGroups.newOrganizationsHelp",
              )}
            </p>
          </div>
          <Select
            value={defaultGroupId || "none"}
            disabled={disabled}
            onValueChange={(value) => onDefaultChange(value === "none" ? "" : String(value))}
          >
            <SelectTrigger
              id={`${id}-default-group`}
              aria-describedby={`${id}-default-group-help`}
              className="w-full bg-background font-mono text-xs"
            >
              <SelectValue>
                {defaultIndex >= 0
                  ? groupName(groups[defaultIndex], defaultIndex)
                  : t("customerPortal.createDialog.noLimit")}
              </SelectValue>
            </SelectTrigger>
            <SelectContent>
              {!memberGroups && <SelectItem value="none">{t("customerPortal.createDialog.noLimit")}</SelectItem>}
              {groups.map((group, index) => (
                <SelectItem key={group.id} value={group.id} className="font-mono text-xs">
                  {groupName(group, index)}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      )}
    </section>
  );
}
