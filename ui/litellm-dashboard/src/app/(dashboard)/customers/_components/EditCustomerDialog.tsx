"use client";

import { useState } from "react";
import { toast } from "sonner";
import { $api } from "@/lib/http/api";
import type { EndUser } from "@/app/(dashboard)/hooks/customers/useCustomers";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

interface Props {
  open: boolean;
  onOpenChange: (o: boolean) => void;
  customer: EndUser;
  onSaved: () => void;
}

function EditCustomerForm({ onOpenChange, customer, onSaved }: Omit<Props, "open">) {
  const [alias, setAlias] = useState(customer.alias ?? "");
  const currentBudget = customer.litellm_budget_table?.max_budget;
  const [budget, setBudget] = useState(currentBudget === null || currentBudget === undefined ? "" : String(currentBudget));
  const budgetId = customer.litellm_budget_table?.budget_id ?? null;
  const asText = (n: number | null | undefined) => (n === null || n === undefined ? "" : String(n));
  const [tpm, setTpm] = useState(asText(customer.litellm_budget_table?.tpm_limit));
  const [rpm, setRpm] = useState(asText(customer.litellm_budget_table?.rpm_limit));
  const updateLimits = $api.useMutation("post", "/budget/update");
  const update = $api.useMutation("post", "/customer/update", {
    onSuccess: async () => {
      if (budgetId) {
        await updateLimits.mutateAsync({ body: { budget_id: budgetId, tpm_limit: tpm ? Number(tpm) : null, rpm_limit: rpm ? Number(rpm) : null } });
      }
      toast.success("Customer updated");
      onSaved();
      onOpenChange(false);
    },
    onError: (e) => toast.error(`Failed to update customer: ${String(e)}`),
  });
  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    update.mutate({
      body: {
        user_id: customer.user_id,
        alias: alias.trim() || null,
        blocked: Boolean(customer.blocked),
        max_budget: budget ? Number(budget) : null,
      },
    });
  };
  return (
    <DialogContent className="sm:max-w-md">
      <form onSubmit={submit} className="space-y-4">
        <DialogHeader>
          <DialogTitle>Edit customer</DialogTitle>
          <DialogDescription><span className="font-mono text-xs">{customer.user_id}</span></DialogDescription>
        </DialogHeader>
        <div className="space-y-1.5">
          <Label htmlFor="ec-alias">Alias</Label>
          <Input id="ec-alias" value={alias} onChange={(e) => setAlias(e.target.value)} autoFocus />
        </div>
        <div className="space-y-1.5">
          <Label htmlFor="ec-budget">Max Budget (USD)</Label>
          <Input id="ec-budget" inputMode="decimal" value={budget} onChange={(e) => setBudget(e.target.value.replace(/[^0-9.]/g, ""))} placeholder="Leave empty for no budget" />
        </div>
        <div className="grid grid-cols-2 gap-3">
          <div className="space-y-1.5">
            <Label htmlFor="ec-tpm">TPM limit</Label>
            <Input id="ec-tpm" inputMode="numeric" value={tpm} onChange={(e) => setTpm(e.target.value.replace(/[^0-9]/g, ""))} placeholder="No limit" disabled={!budgetId} />
          </div>
          <div className="space-y-1.5">
            <Label htmlFor="ec-rpm">RPM limit</Label>
            <Input id="ec-rpm" inputMode="numeric" value={rpm} onChange={(e) => setRpm(e.target.value.replace(/[^0-9]/g, ""))} placeholder="No limit" disabled={!budgetId} />
          </div>
        </div>
        <p className="text-xs text-muted-foreground">{budgetId ? "Limits apply to this customer across every key it uses." : "Set a budget or limits when creating the customer to enable TPM/RPM here."}</p>
        <DialogFooter>
          <Button type="button" variant="outline" onClick={() => onOpenChange(false)}>Cancel</Button>
          <Button type="submit" disabled={update.isPending || updateLimits.isPending}>Save</Button>
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
