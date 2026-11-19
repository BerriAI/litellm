import { useId } from "react";
import { Plus, X } from "lucide-react";
import { useTranslation } from "react-i18next";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import type { CustomerBudgetGroupFields } from "./customerOrganizationForm";
import CustomerBudgetAmountInput from "./CustomerBudgetAmountInput";
import { CustomerRateLimitInput, rateFields } from "./CustomerRateLimits";

const columns = [
  ["groupColumn", "w-[21%]", "w-[26%]"],
  ["monthlyColumn", "w-[13%]", "w-[18%]"],
  ["yearlyColumn", "w-[13%]", "w-[18%]"],
  ["tpmColumn", "w-[12%]", "w-[16%]"],
  ["rpmColumn", "w-[12%]", "w-[16%]"],
] as const;
const budgetFields = [
  ["monthlyBudget", "monthly"],
  ["yearlyBudget", "yearly"],
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
  scope: "member" | "organization" | "app";
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
        <Table className={`${memberGroups ? "min-w-[60rem]" : "min-w-[48rem]"} table-fixed`} aria-label={title}>
          <TableHeader className="bg-muted/40">
            <TableRow className="hover:bg-transparent">
              {columns.map(([key, memberWidth, globalWidth]) => (
                <TableHead
                  key={key}
                  scope="col"
                  className={`${memberGroups ? memberWidth : globalWidth} px-3 text-xs text-muted-foreground`}
                >
                  {t(`customerPortal.createDialog.${key}`)}
                </TableHead>
              ))}
              {memberGroups && (
                <TableHead scope="col" className="w-[23%] px-3 text-xs text-muted-foreground">
                  {t("customerPortal.createDialog.capColumn")}
                </TableHead>
              )}
              <TableHead className="w-[6%]">
                <span className="sr-only">{t("customerPortal.createDialog.groupActions")}</span>
              </TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {groups.map((group, index) => (
              <TableRow key={group.id} className="hover:bg-transparent">
                <TableCell className="p-2 pl-3">
                  <Input
                    aria-label={t("customerPortal.createDialog.groupNameLabel", { index: index + 1 })}
                    value={group.name}
                    onChange={(event) => onUpdate(group.id, { name: event.target.value })}
                    className="font-mono text-xs"
                  />
                </TableCell>
                {budgetFields.map(([field, period]) => (
                  <TableCell key={period} className="p-2">
                    <CustomerBudgetAmountInput
                      id={`${id}-${period}-${group.id}`}
                      label={t(`customerPortal.createDialog.${period}Label`, { index: index + 1 })}
                      value={group[field]}
                      onChange={(value) => onUpdate(group.id, { [field]: value })}
                    />
                  </TableCell>
                ))}
                {rateFields.map((field) => (
                  <TableCell key={field} className="p-2">
                    <CustomerRateLimitInput
                      aria-label={t(`customerPortal.createDialog.${field}Label`, { index: index + 1 })}
                      value={group[field]}
                      onChange={(event) => onUpdate(group.id, { [field]: event.target.value })}
                    />
                  </TableCell>
                ))}
                {memberGroups && (
                  <TableCell className="p-2">
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
                        <SelectValue>
                          {t(
                            group.capPolicy === "pause_allow_increase"
                              ? "customerPortal.createDialog.pauseRequest"
                              : "customerPortal.createDialog.pause",
                          )}
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
                <TableCell className="p-2 pr-3">
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
      {scope !== "app" && (
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
