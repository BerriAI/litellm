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

interface Props {
  open: boolean;
  onOpenChange: (o: boolean) => void;
  onCreated: (customerOrgId: string) => void;
}

const errorText = (e: unknown) => (e instanceof Error ? e.message : String(e));

export default function CreateCustomerDialog({ open, onOpenChange, onCreated }: Props) {
  const { t } = useTranslation();
  const [customerOrgId, setCustomerOrgId] = useState("");
  const [alias, setAlias] = useState("");
  const create = $api.useMutation("post", "/management/v1/customer_organizations", {
    onSuccess: (customer) => {
      toast.success(t("customerPortal.toasts.created", { name: alias.trim() || customerOrgId.trim() }));
      onCreated(customer.customer_org_id);
      setCustomerOrgId("");
      setAlias("");
      onOpenChange(false);
    },
    onError: (e: unknown) => toast.error(t("customerPortal.toasts.createFailed", { error: errorText(e) })),
  });
  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    if (customerOrgId.trim()) {
      create.mutate({
        body: { customer_org_id: customerOrgId.trim(), alias: alias.trim() || null, default_budgets: [] },
      });
    }
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
            <Label htmlFor="cc-org-id">{t("customerPortal.createDialog.customerId")}</Label>
            <Input
              id="cc-org-id"
              value={customerOrgId}
              onChange={(e) => setCustomerOrgId(e.target.value)}
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
          <DialogFooter>
            <Button type="button" variant="outline" onClick={() => onOpenChange(false)}>
              {t("customerPortal.common.cancel")}
            </Button>
            <Button type="submit" disabled={!customerOrgId.trim() || create.isPending}>
              {t("customerPortal.createDialog.submit")}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}
