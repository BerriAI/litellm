"use client";

import { useId, useState } from "react";
import { useTranslation } from "react-i18next";
import { toast } from "sonner";
import { $api } from "@/lib/http/api";
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
  buildCreateCustomerBody,
  decimalOnly,
  hasIncompleteCustomerFallbacks,
  hasInvalidCustomerRateLimits,
  type CustomerFallbackFields as CustomerFallbackRuleFields,
  type ModelRateLimitFields,
} from "./customerForms";
import CustomerRulesFields from "./CustomerRulesFields";

interface Props {
  open: boolean;
  onOpenChange: (o: boolean) => void;
  onCreated: (userId: string) => void;
}

const errorText = (e: unknown) => (e instanceof Error ? e.message : String(e));

export default function CreateCustomerDialog({ open, onOpenChange, onCreated }: Props) {
  const { t } = useTranslation();
  const rateLimitId = useId();
  const [userId, setUserId] = useState("");
  const [alias, setAlias] = useState("");
  const [budget, setBudget] = useState("");
  const [tpm, setTpm] = useState("");
  const [rpm, setRpm] = useState("");
  const [modelLimits, setModelLimits] = useState<ModelRateLimitFields[]>([]);
  const [fallbacks, setFallbacks] = useState<CustomerFallbackRuleFields[] | null>(null);
  const invalidLimits = hasInvalidCustomerRateLimits({ tpm, rpm, modelLimits });
  const create = $api.useMutation("post", "/customer/new", {
    onSuccess: () => {
      toast.success(t("customerPortal.toasts.created", { name: alias.trim() || userId.trim() }));
      onCreated(userId.trim());
      onOpenChange(false);
    },
    onError: (e: unknown) => toast.error(t("customerPortal.toasts.createFailed", { error: errorText(e) })),
  });
  const invalidRules = invalidLimits || hasIncompleteCustomerFallbacks(fallbacks);
  const submitDisabled = !userId.trim() || create.isPending || invalidRules;
  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    const fields = { userId, alias, budget, tpm, rpm, modelLimits, fallbacks };
    const body = buildCreateCustomerBody(fields);
    if (body) create.mutate({ body });
  };
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-h-[90dvh] overflow-y-auto sm:max-w-3xl">
        <form onSubmit={submit} className="space-y-4">
          <DialogHeader>
            <DialogTitle>{t("customerPortal.createDialog.title")}</DialogTitle>
            <DialogDescription>{t("customerPortal.createDialog.description")}</DialogDescription>
          </DialogHeader>
          <div className="grid gap-3 sm:grid-cols-2">
            <div className="space-y-2">
              <Label htmlFor="cc-user-id" className="leading-5">
                {t("customerPortal.createDialog.customerId")}
              </Label>
              <Input
                id="cc-user-id"
                className="h-10"
                value={userId}
                onChange={(e) => setUserId(e.target.value)}
                placeholder={t("customerPortal.createDialog.customerIdPlaceholder")}
                autoFocus
                required
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="cc-alias" className="leading-5">
                {t("customerPortal.createDialog.alias")}
              </Label>
              <Input
                id="cc-alias"
                className="h-10"
                value={alias}
                onChange={(e) => setAlias(e.target.value)}
                placeholder={t("customerPortal.createDialog.aliasPlaceholder")}
              />
            </div>
          </div>
          <div className="space-y-2">
            <Label htmlFor="cc-budget" className="leading-5">
              {t("customerPortal.createDialog.maxBudget")}
            </Label>
            <Input
              id="cc-budget"
              className="h-10"
              inputMode="decimal"
              value={budget}
              onChange={(e) => setBudget(decimalOnly(e.target.value))}
              placeholder={t("customerPortal.createDialog.maxBudgetPlaceholder")}
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
          <DialogFooter>
            <Button type="button" variant="outline" onClick={() => onOpenChange(false)}>
              {t("customerPortal.common.cancel")}
            </Button>
            <Button type="submit" disabled={submitDisabled}>
              {t("customerPortal.createDialog.submit")}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}
