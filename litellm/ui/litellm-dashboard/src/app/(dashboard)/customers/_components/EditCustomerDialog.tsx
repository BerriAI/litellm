"use client";

import { useId, useState } from "react";
import { useTranslation } from "react-i18next";
import { useMutation } from "@tanstack/react-query";
import { toast } from "sonner";
import { z } from "zod";
import { fetchClient } from "@/lib/http/api";
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
import {
  buildCustomerBudgetBody,
  buildUpdateCustomerBody,
  customerFallbackFields,
  decimalOnly,
  hasIncompleteCustomerFallbacks,
  hasInvalidCustomerRateLimits,
  modelRateLimitFields,
} from "./customerForms";
import CustomerRulesFields from "./CustomerRulesFields";

interface Props {
  open: boolean;
  onOpenChange: (o: boolean) => void;
  customer: EndUser;
  onSaved: () => void;
}

const asText = (n: number | null | undefined) => (n === null || n === undefined ? "" : String(n));
const budgetIdentity = z.object({ budget_id: z.string().min(1) });

function EditCustomerForm({ onOpenChange, customer, onSaved }: Omit<Props, "open">) {
  const { t } = useTranslation();
  const rateLimitId = useId();
  const [alias, setAlias] = useState(customer.alias ?? "");
  const [budget, setBudget] = useState(asText(customer.litellm_budget_table?.max_budget));
  const [tpm, setTpm] = useState(asText(customer.litellm_budget_table?.tpm_limit));
  const [rpm, setRpm] = useState(asText(customer.litellm_budget_table?.rpm_limit));
  const [modelLimits, setModelLimits] = useState(() => modelRateLimitFields(customer.litellm_budget_table));
  const [fallbacks, setFallbacks] = useState(() => customerFallbackFields(customer.fallbacks));
  const [budgetId, setBudgetId] = useState(customer.budget_id ?? customer.litellm_budget_table?.budget_id ?? null);
  const update = useMutation({
    mutationFn: async () => {
      const fields = { userId: customer.user_id, alias, blocked: Boolean(customer.blocked), fallbacks };
      const customerBody = buildUpdateCustomerBody(fields);
      const budgetFields = { budget, tpm, rpm, modelLimits };
      const budgetBody = buildCustomerBudgetBody(budgetFields);
      const hasBudgetValues = [budget, tpm, rpm, ...modelLimits.flatMap((row) => [row.tpm, row.rpm])].some(
        (value) => value.trim() !== "",
      );
      if (budgetId) {
        await fetchClient.POST("/budget/update", { body: { ...budgetBody, budget_id: budgetId } });
        customerBody.budget_id = budgetId;
      } else if (hasBudgetValues) {
        const { data } = await fetchClient.POST("/budget/new", {
          body: { ...budgetBody, ...(budget.trim() && { budget_duration: "1mo" }) },
        });
        const created = budgetIdentity.safeParse(data);
        if (!created.success) {
          throw new Error(t("customerPortal.editDialog.missingBudget"));
        }
        setBudgetId(created.data.budget_id);
        customerBody.budget_id = created.data.budget_id;
      }
      await fetchClient.POST("/customer/update", { body: customerBody });
    },
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
    update.mutate();
  };
  return (
    <DialogContent className="max-h-[90dvh] overflow-y-auto sm:max-w-3xl">
      <form onSubmit={submit} className="space-y-4">
        <DialogHeader>
          <DialogTitle>{t("customerPortal.editDialog.title")}</DialogTitle>
          <DialogDescription>
            <span className="font-mono text-xs">{customer.user_id}</span>
          </DialogDescription>
        </DialogHeader>
        <div className="space-y-2">
          <Label htmlFor="ec-alias" className="h-5 leading-5">
            {t("customerPortal.editDialog.alias")}
          </Label>
          <Input id="ec-alias" className="h-10" value={alias} onChange={(e) => setAlias(e.target.value)} autoFocus />
        </div>
        <div className="space-y-2">
          <Label htmlFor="ec-budget" className="h-5 leading-5">
            {t("customerPortal.editDialog.maxBudget")}
          </Label>
          <Input
            id="ec-budget"
            className="h-10"
            inputMode="decimal"
            value={budget}
            onChange={(e) => setBudget(decimalOnly(e.target.value))}
            placeholder={t("customerPortal.editDialog.maxBudgetPlaceholder")}
          />
        </div>
        <CustomerRulesFields
          idPrefix={rateLimitId}
          tpm={tpm}
          rpm={rpm}
          onTpmChange={setTpm}
          onRpmChange={setRpm}
          modelLimits={modelLimits}
          onModelLimitsChange={setModelLimits}
          fallbacks={fallbacks}
          onFallbacksChange={setFallbacks}
        />
        {budgetId && (
          <p className="text-xs text-muted-foreground">{t("customerPortal.editDialog.sharedBudgetWarning")}</p>
        )}
        <DialogFooter>
          <Button type="button" variant="outline" onClick={() => onOpenChange(false)}>
            {t("customerPortal.common.cancel")}
          </Button>
          <Button
            type="submit"
            disabled={
              update.isPending ||
              hasIncompleteCustomerFallbacks(fallbacks) ||
              hasInvalidCustomerRateLimits({ tpm, rpm, modelLimits })
            }
          >
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
