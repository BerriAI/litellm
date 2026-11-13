"use client";

import { useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { ChevronRight } from "lucide-react";
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
import { Label } from "@/components/ui/label";
import {
  buildCustomerOrganizationBody,
  initialCustomerOrganizationFields,
  type CustomerOrganizationFormError,
} from "./customerOrganizationForm";

import CustomerBudgetGroupsEditor, { BudgetInput } from "./CustomerBudgetGroupsEditor";

interface Props {
  open: boolean;
  onOpenChange: (o: boolean) => void;
  onCreated: (customerOrgId: string) => void;
}

const rateFields = ["tpm", "rpm"] as const;
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
            <CustomerBudgetGroupsEditor
              groups={fields.groups}
              defaultGroupId={fields.defaultGroupId}
              disabled={create.isPending}
              onChange={(groups, defaultGroupId) => {
                setFields({ ...fields, groups, defaultGroupId });
                setError(null);
              }}
            />
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
