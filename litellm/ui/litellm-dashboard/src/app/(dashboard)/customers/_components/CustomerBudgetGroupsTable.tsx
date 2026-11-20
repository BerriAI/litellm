import { useId } from "react";
import { Plus, X } from "lucide-react";
import { useTranslation } from "react-i18next";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import type { CustomerBudgetGroupFields } from "./customerOrganizationForm";
import CustomerAllowedModelsSelect from "./CustomerAllowedModelsSelect";
import CustomerBudgetAmountInput from "./CustomerBudgetAmountInput";
import CustomerBudgetPeriodSelect from "./CustomerBudgetPeriodSelect";
import CustomerBudgetWindows from "./CustomerBudgetWindows";
import { CustomerRateLimitInput, rateFields } from "./CustomerRateLimits";

const columns = [
  ["customerPortal.createDialog.groupColumn", "w-[13%]", "w-[16%]"],
  ["customerPortal.createDialog.spendLimitLabel", "w-[28%]", "w-[28%]"],
  ["customerPortal.createDialog.tpmColumn", "w-[8%]", "w-[10%]"],
  ["customerPortal.createDialog.rpmColumn", "w-[8%]", "w-[10%]"],
  ["customerPortal.hierarchy.softBudget", "w-[9%]", "w-[10%]"],
  ["customerPortal.hierarchy.allowedModels", "w-[17%]", "w-[22%]"],
] as const;

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
  scope: "member" | "organization" | "app" | "overall-app" | "app-member";
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
      <div className="overflow-hidden rounded-lg border">
        <Table className="min-w-[70rem] table-fixed" aria-label={title}>
          <TableHeader className="bg-muted/40">
            <TableRow className="hover:bg-transparent">
              {columns.map(([label, memberWidth, globalWidth]) => (
                <TableHead
                  key={label}
                  scope="col"
                  className={`${memberGroups ? memberWidth : globalWidth} px-3 text-xs font-normal text-muted-foreground`}
                >
                  {t(label)}
                </TableHead>
              ))}
              {memberGroups && (
                <TableHead scope="col" className="w-[13%] px-3 text-xs font-normal text-muted-foreground">
                  {t("customerPortal.createDialog.capColumn")}
                </TableHead>
              )}
              <TableHead className="w-[4%]">
                <span className="sr-only">{t("customerPortal.createDialog.groupActions")}</span>
              </TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {groups.map((group, index) => (
              <TableRow
                key={group.id}
                aria-label={groupName(group, index)}
                className="hover:bg-transparent [&>td]:align-top"
              >
                <TableCell className="pl-3">
                  <Input
                    aria-label={t("customerPortal.createDialog.groupNameLabel", { index: index + 1 })}
                    disabled={disabled}
                    value={group.name}
                    onChange={(event) => onUpdate(group.id, { name: event.target.value })}
                    className="font-mono text-xs"
                  />
                </TableCell>
                <TableCell>
                  <div className="flex items-center gap-1.5 overflow-x-auto">
                    <CustomerBudgetAmountInput
                      id={`${id}-budget-${group.id}`}
                      label={t("customerPortal.createDialog.spendLimitLabel")}
                      disabled={disabled}
                      value={group.maxBudget}
                      onChange={(maxBudget) => onUpdate(group.id, { maxBudget })}
                      compact
                    >
                      <CustomerBudgetPeriodSelect
                        value={group.budgetDuration}
                        disabled={disabled}
                        onChange={(budgetDuration) => onUpdate(group.id, { budgetDuration })}
                        unavailablePeriods={group.budgetWindows
                          .filter(({ maxBudget }) => maxBudget.trim() !== "")
                          .map(({ budget }) => budget.budget_duration)}
                        compact
                      />
                    </CustomerBudgetAmountInput>
                    <CustomerBudgetWindows
                      id={`${id}-${group.id}`}
                      windows={group.budgetWindows}
                      onChange={(budgetWindows) => onUpdate(group.id, { budgetWindows })}
                      primaryPeriod={group.maxBudget.trim() ? group.budgetDuration : undefined}
                      onAdd={(limit) =>
                        onUpdate(
                          group.id,
                          group.maxBudget.trim()
                            ? { budgetWindows: [...group.budgetWindows, limit] }
                            : { maxBudget: limit.maxBudget, budgetDuration: limit.budget.budget_duration },
                        )
                      }
                      disabled={disabled}
                      compact
                    />
                  </div>
                </TableCell>
                {rateFields.map((field) => (
                  <TableCell key={field}>
                    <CustomerRateLimitInput
                      aria-label={t(`customerPortal.hierarchy.${field}`)}
                      disabled={disabled}
                      value={group[field]}
                      onChange={(event) => onUpdate(group.id, { [field]: event.target.value })}
                    />
                  </TableCell>
                ))}
                <TableCell>
                  <CustomerBudgetAmountInput
                    id={`${id}-soft-${group.id}`}
                    label={t("customerPortal.hierarchy.softBudget")}
                    disabled={disabled}
                    value={group.softBudget}
                    onChange={(softBudget) => onUpdate(group.id, { softBudget })}
                  />
                </TableCell>
                <TableCell>
                  <CustomerAllowedModelsSelect
                    id={`${id}-models-${group.id}`}
                    value={group.allowedModels}
                    onChange={(allowedModels) => onUpdate(group.id, { allowedModels })}
                    disabled={disabled}
                    showHint={false}
                  />
                </TableCell>
                {memberGroups && (
                  <TableCell>
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
                        aria-label={t("customerPortal.budgetGroups.atCapLabel", { index: index + 1 })}
                        className="w-full text-xs"
                      >
                        <SelectValue className="min-w-0">
                          <span className="truncate">
                            {t(
                              group.capPolicy === "pause_allow_increase"
                                ? "customerPortal.createDialog.pauseRequest"
                                : "customerPortal.createDialog.pause",
                            )}
                          </span>
                        </SelectValue>
                      </SelectTrigger>
                      <SelectContent align="end" className="w-80 max-w-[calc(100vw-2rem)] p-1">
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
                  </TableCell>
                )}
                <TableCell className="px-1">
                  <Button
                    type="button"
                    variant="ghost"
                    size="icon-sm"
                    disabled={disabled || (memberGroups && groups.length === 1)}
                    onClick={() => onRemove(group.id)}
                    aria-label={t("customerPortal.createDialog.removeGroup", { name: groupName(group, index) })}
                    className="text-muted-foreground"
                  >
                    <X className="size-3.5" />
                  </Button>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
        <div className="border-t p-2">
          <Button type="button" variant="ghost" size="sm" disabled={disabled} onClick={onAdd}>
            <Plus className="size-4" />
            {t("customerPortal.createDialog.addGroup")}
          </Button>
        </div>
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
