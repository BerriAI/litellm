"use client";

import { Controller } from "react-hook-form";
import { useTranslation } from "react-i18next";
import { toast } from "sonner";
import { $api } from "@/lib/http/api";
import { useZodForm } from "@/lib/forms/useZodForm";
import type { EndUser } from "@/app/(dashboard)/hooks/customers/useCustomers";
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
import { decimalOnly } from "./customerForms";
import { customerToForm, toUpdateCustomerBody } from "./editCustomerMapper";
import { editCustomerSchema } from "./editCustomerSchema";

interface Props {
  open: boolean;
  onOpenChange: (o: boolean) => void;
  customer: EndUser;
  onSaved: () => void;
}

function EditCustomerForm({ onOpenChange, customer, onSaved }: Omit<Props, "open">) {
  const { t } = useTranslation();
  const form = useZodForm(editCustomerSchema, { defaultValues: customerToForm(customer) });
  const budgetId = customer.litellm_budget_table?.budget_id ?? null;
  const update = $api.useMutation("post", "/customer/update", {
    onSuccess: () => {
      toast.success(t("customerPortal.toasts.updated"));
      onSaved();
      onOpenChange(false);
    },
    onError: (e: unknown) =>
      toast.error(t("customerPortal.toasts.updateFailed", { error: e instanceof Error ? e.message : String(e) })),
  });
  const onSubmit = form.handleSubmit((values) => update.mutate({ body: toUpdateCustomerBody(customer, values) }));
  return (
    <DialogContent className="sm:max-w-md">
      <form onSubmit={onSubmit} className="space-y-4">
        <DialogHeader>
          <DialogTitle>{t("customerPortal.editDialog.title")}</DialogTitle>
          <DialogDescription>
            <span className="font-mono text-xs">{customer.user_id}</span>
          </DialogDescription>
        </DialogHeader>
        <div className="space-y-1.5">
          <Label htmlFor="ec-alias">{t("customerPortal.editDialog.alias")}</Label>
          <Controller
            control={form.control}
            name="alias"
            render={({ field }) => (
              <Input
                id="ec-alias"
                ref={field.ref}
                value={field.value}
                onChange={field.onChange}
                onBlur={field.onBlur}
                autoFocus
              />
            )}
          />
        </div>
        <div className="space-y-1.5">
          <Label htmlFor="ec-budget">{t("customerPortal.editDialog.maxBudget")}</Label>
          <Controller
            control={form.control}
            name="budget"
            render={({ field }) => (
              <Input
                id="ec-budget"
                ref={field.ref}
                inputMode="decimal"
                value={field.value}
                onChange={(e) => field.onChange(decimalOnly(e.target.value))}
                onBlur={field.onBlur}
                placeholder={t("customerPortal.editDialog.maxBudgetPlaceholder")}
              />
            )}
          />
        </div>
        {budgetId && (
          <p className="text-xs text-muted-foreground">
            {t("customerPortal.editDialog.sharedBudgetWarning", { budgetId })}
          </p>
        )}
        <DialogFooter>
          <Button type="button" variant="outline" onClick={() => onOpenChange(false)}>
            {t("customerPortal.common.cancel")}
          </Button>
          <Button type="submit" disabled={update.isPending}>
            {t("customerPortal.common.save")}
          </Button>
        </DialogFooter>
      </form>
    </DialogContent>
  );
}

export default function EditCustomerDialog({ open, ...form }: Props) {
  return (
    <Dialog open={open} onOpenChange={form.onOpenChange}>
      {open && <EditCustomerForm {...form} />}
    </Dialog>
  );
}
