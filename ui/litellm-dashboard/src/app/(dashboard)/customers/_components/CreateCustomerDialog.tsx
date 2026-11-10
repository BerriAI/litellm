"use client";

import { useState } from "react";
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
import { buildCreateCustomerBody, digitsOnly, decimalOnly } from "./customerForms";

interface Props {
  open: boolean;
  onOpenChange: (o: boolean) => void;
  onCreated: (userId: string) => void;
}

const errorText = (e: unknown) => (e instanceof Error ? e.message : String(e));

export default function CreateCustomerDialog({ open, onOpenChange, onCreated }: Props) {
  const { t } = useTranslation();
  const [userId, setUserId] = useState("");
  const [alias, setAlias] = useState("");
  const [budget, setBudget] = useState("");
  const [tpm, setTpm] = useState("");
  const [rpm, setRpm] = useState("");
  const create = $api.useMutation("post", "/customer/new", {
    onSuccess: () => {
      toast.success(t("customerPortal.toasts.created", { name: alias.trim() || userId.trim() }));
      onCreated(userId.trim());
      onOpenChange(false);
    },
    onError: (e: unknown) => toast.error(t("customerPortal.toasts.createFailed", { error: errorText(e) })),
  });
  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    const fields = { userId, alias, budget, tpm, rpm };
    const body = buildCreateCustomerBody(fields);
    if (body) create.mutate({ body });
  };
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-md">
        <form onSubmit={submit} className="space-y-4">
          <DialogHeader>
            <DialogTitle>{t("customerPortal.createDialog.title")}</DialogTitle>
            <DialogDescription>{t("customerPortal.createDialog.description")}</DialogDescription>
          </DialogHeader>
          <div className="space-y-1.5">
            <Label htmlFor="cc-user-id">{t("customerPortal.createDialog.customerId")}</Label>
            <Input
              id="cc-user-id"
              value={userId}
              onChange={(e) => setUserId(e.target.value)}
              placeholder={t("customerPortal.createDialog.customerIdPlaceholder")}
              autoFocus
              required
            />
          </div>
          <div className="space-y-1.5">
            <Label htmlFor="cc-alias">{t("customerPortal.createDialog.alias")}</Label>
            <Input
              id="cc-alias"
              value={alias}
              onChange={(e) => setAlias(e.target.value)}
              placeholder={t("customerPortal.createDialog.aliasPlaceholder")}
            />
          </div>
          <div className="space-y-1.5">
            <Label htmlFor="cc-budget">{t("customerPortal.createDialog.maxBudget")}</Label>
            <Input
              id="cc-budget"
              inputMode="decimal"
              value={budget}
              onChange={(e) => setBudget(decimalOnly(e.target.value))}
              placeholder={t("customerPortal.createDialog.maxBudgetPlaceholder")}
            />
          </div>
          <div className="grid grid-cols-2 gap-3">
            <div className="space-y-1.5">
              <Label htmlFor="cc-tpm">{t("customerPortal.createDialog.tpm")}</Label>
              <Input
                id="cc-tpm"
                inputMode="numeric"
                value={tpm}
                onChange={(e) => setTpm(digitsOnly(e.target.value))}
                placeholder={t("customerPortal.createDialog.noLimit")}
              />
            </div>
            <div className="space-y-1.5">
              <Label htmlFor="cc-rpm">{t("customerPortal.createDialog.rpm")}</Label>
              <Input
                id="cc-rpm"
                inputMode="numeric"
                value={rpm}
                onChange={(e) => setRpm(digitsOnly(e.target.value))}
                placeholder={t("customerPortal.createDialog.noLimit")}
              />
            </div>
          </div>
          <p className="text-xs text-muted-foreground">{t("customerPortal.createDialog.limitsHelp")}</p>
          <DialogFooter>
            <Button type="button" variant="outline" onClick={() => onOpenChange(false)}>
              {t("customerPortal.common.cancel")}
            </Button>
            <Button type="submit" disabled={!userId.trim() || create.isPending}>
              {t("customerPortal.createDialog.submit")}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}
