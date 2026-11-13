"use client";

import { useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { ChevronRight, Plus, Trash2 } from "lucide-react";
import { useTranslation } from "react-i18next";
import { toast } from "sonner";
import { $api } from "@/lib/http/api";
import { extractProxyErrorMessage } from "@/lib/http/client";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { InputGroup, InputGroupAddon, InputGroupInput, InputGroupText } from "@/components/ui/input-group";
import { Label } from "@/components/ui/label";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import {
  buildCustomerOrganizationBody,
  initialCustomerOrganizationFields,
  type CustomerBudgetGroupFields,
  type CustomerOrganizationFormError,
} from "./customerOrganizationForm";

interface Props {
  open: boolean;
  onOpenChange: (o: boolean) => void;
  onCreated: (customerOrgId: string) => void;
}

function BudgetInput({
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

const rateFields = ["tpm", "rpm"] as const;
const groupBudgetFields = [
  ["monthlyBudget", "monthly"],
  ["yearlyBudget", "yearly"],
] as const;

export default function CreateCustomerDialog({ open, onOpenChange, onCreated }: Props) {
  const queryClient = useQueryClient();
  const { t } = useTranslation();
  const [fields, setFields] = useState(initialCustomerOrganizationFields);
  const [error, setError] = useState<CustomerOrganizationFormError | null>(null);
  const create = $api.useMutation("post", "/management/v1/customer_organizations", {
    onSettled: () => queryClient.invalidateQueries({ queryKey: ["get", "/management/v1/customer_portal/limits"] }),
    onSuccess: (customer) => {
      toast.success(t("customerPortal.toasts.created", { name: customer.alias || fields.name.trim() }));
      onCreated(customer.customer_org_id);
      setFields(initialCustomerOrganizationFields());
      setError(null);
      onOpenChange(false);
    },
    onError: (e: unknown) =>
      toast.error(t("customerPortal.toasts.createFailed", { error: extractProxyErrorMessage(e) })),
  });
  const updateGroup = (id: string, patch: Partial<Omit<CustomerBudgetGroupFields, "id">>) => {
    setFields((current) => ({
      ...current,
      groups: current.groups.map((group) => (group.id === id ? { ...group, ...patch } : group)),
    }));
    setError(null);
  };
  const removeGroup = (id: string) => {
    setFields((current) => {
      const groups = current.groups.filter((group) => group.id !== id);
      return {
        ...current,
        groups,
        defaultGroupId: current.defaultGroupId === id ? groups[0].id : current.defaultGroupId,
      };
    });
    setError(null);
  };
  const submit = (event: React.FormEvent) => {
    event.preventDefault();
    if (create.isPending) return;
    const result = buildCustomerOrganizationBody(fields);
    if (!result.ok) {
      setError(result.error);
      return;
    }
    setError(null);
    create.mutate({ body: result.body });
  };
  return (
    <Dialog open={open} onOpenChange={(next) => !create.isPending && onOpenChange(next)}>
      <DialogContent className="max-h-[90dvh] overflow-y-auto sm:max-w-2xl">
        <form onSubmit={submit} className="space-y-5">
          <DialogHeader>
            <DialogTitle>{t("customerPortal.createDialog.title")}</DialogTitle>
            <DialogDescription>{t("customerPortal.createDialog.description")}</DialogDescription>
          </DialogHeader>
          <fieldset disabled={create.isPending} className="space-y-5">
            <div className="grid gap-4 sm:grid-cols-2">
              <div className="space-y-1.5">
                <Label htmlFor="cc-name">{t("customerPortal.createDialog.name")}</Label>
                <Input
                  id="cc-name"
                  value={fields.name}
                  onChange={(event) => setFields({ ...fields, name: event.target.value })}
                  placeholder={t("customerPortal.createDialog.namePlaceholder")}
                  autoFocus
                  required
                />
              </div>
              <div className="space-y-1.5">
                <Label htmlFor="cc-org-id">{t("customerPortal.createDialog.customerId")}</Label>
                <Input
                  id="cc-org-id"
                  value={fields.customerOrgId}
                  onChange={(event) => setFields({ ...fields, customerOrgId: event.target.value })}
                  placeholder={t("customerPortal.createDialog.customerIdPlaceholder")}
                  aria-describedby="cc-org-id-help"
                />
                <p id="cc-org-id-help" className="text-xs text-muted-foreground">
                  {t("customerPortal.createDialog.customerIdHelp")}
                </p>
              </div>
            </div>
            <div className="space-y-1.5">
              <Label htmlFor="cc-budget">{t("customerPortal.createDialog.maxBudget")}</Label>
              <BudgetInput
                id="cc-budget"
                label={t("customerPortal.createDialog.maxBudget")}
                value={fields.monthlyBudget}
                onChange={(monthlyBudget) => setFields({ ...fields, monthlyBudget })}
              />
              <p className="text-xs text-muted-foreground">{t("customerPortal.createDialog.budgetHelp")}</p>
            </div>
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
                value={fields.defaultGroupId}
                onValueChange={(defaultGroupId) => setFields({ ...fields, defaultGroupId: String(defaultGroupId) })}
                aria-label={t("customerPortal.createDialog.defaultGroup")}
                disabled={create.isPending}
                className="gap-1"
              >
                {fields.groups.map((group, index) => (
                  <div
                    key={group.id}
                    className={`grid grid-cols-2 items-center gap-3 rounded-md p-2 sm:grid-cols-[1.3fr_1fr_1fr_3rem_2rem] ${group.id === fields.defaultGroupId ? "bg-muted/60" : ""}`}
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
                    <Label className="flex gap-2 sm:justify-center">
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
                      className="justify-self-end"
                      disabled={fields.groups.length === 1}
                      onClick={() => removeGroup(group.id)}
                      aria-label={t("customerPortal.createDialog.removeGroup", { name: group.name || index + 1 })}
                    >
                      <Trash2 className="size-3.5" />
                    </Button>
                  </div>
                ))}
              </RadioGroup>
              <Button
                type="button"
                variant="ghost"
                size="sm"
                onClick={() =>
                  setFields({
                    ...fields,
                    groups: [
                      ...fields.groups,
                      { id: crypto.randomUUID(), name: "", monthlyBudget: "", yearlyBudget: "" },
                    ],
                  })
                }
              >
                <Plus className="size-4" />
                {t("customerPortal.createDialog.addGroup")}
              </Button>
              <p className="text-xs text-muted-foreground">{t("customerPortal.createDialog.defaultHelp")}</p>
              <p className="text-xs text-muted-foreground">{t("customerPortal.createDialog.resetHelp")}</p>
            </section>
            <details className="group border-t pt-4">
              <summary className="flex cursor-pointer list-none items-center gap-1 text-sm font-medium">
                <ChevronRight className="size-4 group-open:rotate-90" />
                {t("customerPortal.createDialog.advanced")}
              </summary>
              <div className="mt-3 grid grid-cols-2 gap-3">
                {rateFields.map((field) => (
                  <div className="space-y-1.5" key={field}>
                    <Label htmlFor={`cc-${field}`}>{t(`customerPortal.createDialog.${field}`)}</Label>
                    <Input
                      id={`cc-${field}`}
                      type="number"
                      min={0}
                      step={1}
                      value={fields[field]}
                      onChange={(event) => setFields({ ...fields, [field]: event.target.value })}
                      placeholder={t("customerPortal.createDialog.noLimit")}
                    />
                  </div>
                ))}
              </div>
            </details>
          </fieldset>
          {error && (
            <p role="alert" className="text-sm text-destructive">
              {t(`customerPortal.createDialog.errors.${error}`)}
            </p>
          )}
          <DialogFooter>
            <Button type="button" variant="outline" onClick={() => onOpenChange(false)} disabled={create.isPending}>
              {t("customerPortal.common.cancel")}
            </Button>
            <Button type="submit" disabled={!fields.name.trim() || create.isPending}>
              {create.isPending ? t("customerPortal.createDialog.creating") : t("customerPortal.createDialog.submit")}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}
