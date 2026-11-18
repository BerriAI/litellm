"use client";

import { useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { toast } from "sonner";
import { $api } from "@/lib/http/api";
import { extractProxyErrorMessage } from "@/lib/http/client";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  buildCustomerOrganizationBody,
  initialCustomerOrganizationFields,
  type CustomerOrganizationFormError,
} from "./customerOrganizationForm";

import CustomerBudgetInput from "./CustomerBudgetInput";
import CustomerBudgetGroupsEditor from "./CustomerBudgetGroupsEditor";
import { CustomerRateLimitFields } from "./CustomerRateLimits";

interface Props {
  open: boolean;
  onOpenChange: (o: boolean) => void;
  onCreated: (customerOrgId: string) => void;
}

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
            <CustomerBudgetInput
              id="cc-budget"
              budgetId={fields.budgetId}
              amount={fields.monthlyBudget}
              onChange={(budgetId, monthlyBudget) => setFields({ ...fields, budgetId, monthlyBudget })}
            />
            {!fields.budgetId && (
              <CustomerRateLimitFields
                id="cc"
                tpm={fields.tpm}
                rpm={fields.rpm}
                onChange={(field, value) => setFields({ ...fields, [field]: value })}
              />
            )}
            <CustomerBudgetGroupsEditor
              groups={fields.groups}
              defaultGroupId={fields.defaultGroupId}
              disabled={create.isPending}
              onChange={(groups, defaultGroupId) => {
                setFields({ ...fields, groups, defaultGroupId });
                setError(null);
              }}
            />
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
