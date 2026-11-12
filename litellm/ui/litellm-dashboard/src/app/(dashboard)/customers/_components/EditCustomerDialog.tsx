"use client";

import { useState } from "react";
import { useTranslation } from "react-i18next";
import { toast } from "sonner";
import { $api } from "@/lib/http/api";
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
import { buildUpdateCustomerBody, decimalOnly } from "./customerForms";

interface Props {
  open: boolean;
  onOpenChange: (o: boolean) => void;
  customer: EndUser;
  onSaved: () => void;
}

const asText = (n: number | null | undefined) => (n === null || n === undefined ? "" : String(n));

function EditCustomerForm({ onOpenChange, customer, onSaved }: Omit<Props, "open">) {
  const { t } = useTranslation();
  const [alias, setAlias] = useState(customer.alias ?? "");
  const [budget, setBudget] = useState(asText(customer.litellm_budget_table?.max_budget));
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
  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    const fields = { userId: customer.user_id, alias, budget, blocked: Boolean(customer.blocked) };
    update.mutate({ body: buildUpdateCustomerBody(fields) });
  };
  return (
    <DialogContent className="sm:max-w-md">
      <form onSubmit={submit} className="space-y-4">
        <DialogHeader>
          <DialogTitle>{t("customerPortal.editDialog.title")}</DialogTitle>
          <DialogDescription>
            <span className="font-mono text-xs">{customer.user_id}</span>
          </DialogDescription>
        </DialogHeader>
        <div className="space-y-1.5">
          <Label htmlFor="ec-alias">{t("customerPortal.editDialog.alias")}</Label>
          <Input id="ec-alias" value={alias} onChange={(e) => setAlias(e.target.value)} autoFocus />
        </div>
        <div className="space-y-1.5">
          <Label htmlFor="ec-budget">{t("customerPortal.editDialog.maxBudget")}</Label>
          <Input
            id="ec-budget"
            inputMode="decimal"
            value={budget}
            onChange={(e) => setBudget(decimalOnly(e.target.value))}
            placeholder={t("customerPortal.editDialog.maxBudgetPlaceholder")}
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
