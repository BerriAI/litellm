"use client";

import { useState } from "react";
import { toast } from "sonner";
import { $api } from "@/lib/http/api";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

export default function CreateCustomerDialog({ open, onOpenChange, onCreated }: { open: boolean; onOpenChange: (o: boolean) => void; onCreated: (userId: string) => void }) {
  const [userId, setUserId] = useState("");
  const [alias, setAlias] = useState("");
  const [budget, setBudget] = useState("");
  const create = $api.useMutation("post", "/customer/new", {
    onSuccess: () => {
      toast.success(`Created ${alias || userId}`);
      onCreated(userId.trim());
      onOpenChange(false);
    },
    onError: (e: unknown) => toast.error(`Failed to create customer: ${e instanceof Error ? e.message : String(e)}`),
  });
  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!userId.trim()) return;
    create.mutate({
      body: {
        user_id: userId.trim(),
        alias: alias.trim() || null,
        blocked: false,
        max_budget: budget ? Number(budget) : null,
        budget_duration: budget ? "1mo" : null,
      },
    });
  };
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-lg">
        <form onSubmit={submit} className="space-y-4">
          <DialogHeader>
            <DialogTitle>Create customer</DialogTitle>
            <DialogDescription>Track spend and set limits for an end user of your product.</DialogDescription>
          </DialogHeader>
          <div className="grid gap-4 sm:grid-cols-2">
            <div className="space-y-1.5">
              <Label htmlFor="cp-id">Customer ID</Label>
              <Input id="cp-id" value={userId} onChange={(e) => setUserId(e.target.value)} placeholder="cust_acme_prod" autoFocus />
              <p className="text-xs text-muted-foreground">Sent as the <code>user</code> field or <code>x-litellm-customer-id</code> header.</p>
            </div>
            <div className="space-y-1.5">
              <Label htmlFor="cp-alias">Alias</Label>
              <Input id="cp-alias" value={alias} onChange={(e) => setAlias(e.target.value)} placeholder="Acme Corp" />
            </div>
          </div>
          <div className="space-y-1.5 sm:w-1/2">
            <Label htmlFor="cp-budget">Max Budget (USD, resets monthly)</Label>
            <Input id="cp-budget" inputMode="decimal" value={budget} onChange={(e) => setBudget(e.target.value.replace(/[^0-9.]/g, ""))} placeholder="Leave empty for no budget" />
          </div>
          <DialogFooter>
            <Button type="button" variant="outline" onClick={() => onOpenChange(false)}>Cancel</Button>
            <Button type="submit" disabled={!userId.trim() || create.isPending}>Create Customer</Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}
