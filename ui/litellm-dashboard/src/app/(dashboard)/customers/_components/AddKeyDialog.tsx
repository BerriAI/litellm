"use client";

import { useState } from "react";
import Link from "next/link";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { uiHref } from "@/utils/uiHref";
import { setCustomerTag, type PortalKey } from "./customerKeys";

interface Props {
  open: boolean;
  onOpenChange: (o: boolean) => void;
  accessToken: string;
  customerId: string;
  customerLabel: string;
  keys: PortalKey[];
  excludeTokens: ReadonlySet<string>;
  onSaved: () => void;
}

function AddKeyForm({ onOpenChange, accessToken, customerId, customerLabel, keys, excludeTokens, onSaved }: Omit<Props, "open">) {
  const available = keys.filter((k) => !excludeTokens.has(k.token));
  const [token, setToken] = useState(available[0]?.token ?? "");
  const [saving, setSaving] = useState(false);
  const selected = available.find((k) => k.token === token) ?? null;

  const save = async () => {
    if (!selected) return;
    setSaving(true);
    try {
      await setCustomerTag(accessToken, selected, customerId, true);
      toast.success(`Added ${selected.alias} to ${customerLabel}`);
      onSaved();
      onOpenChange(false);
    } catch (e) {
      toast.error(`Failed to update key: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setSaving(false);
    }
  };

  return (
    <DialogContent className="sm:max-w-md">
      <DialogHeader>
        <DialogTitle>Add key to {customerLabel}</DialogTitle>
        <DialogDescription>Tags the key with this customer id so it shows up here. Requests from this key should send <code>user={customerId}</code>.</DialogDescription>
      </DialogHeader>
      <div className="space-y-1.5">
        <Label htmlFor="ak-key">Key</Label>
        <Select value={token} onValueChange={(v) => setToken(v ?? "")}>
          <SelectTrigger id="ak-key" className="w-full"><SelectValue placeholder="Select a key">{selected?.alias}</SelectValue></SelectTrigger>
          <SelectContent>
            {available.map((k) => (
              <SelectItem key={k.token} value={k.token}>{k.alias}</SelectItem>
            ))}
          </SelectContent>
        </Select>
        <p className="text-xs text-muted-foreground">
          Don&apos;t see the key? <Link id="ak-create-link" href={uiHref("api-keys")} className="font-medium text-primary hover:underline">Create one in AI Gateway</Link>
        </p>
      </div>
      <DialogFooter>
        <Button type="button" variant="outline" onClick={() => onOpenChange(false)}>Cancel</Button>
        <Button type="button" disabled={!selected || saving} onClick={save}>Add key</Button>
      </DialogFooter>
    </DialogContent>
  );
}

export default function AddKeyDialog({ open, ...form }: Props) {
  return (
    <Dialog open={open} onOpenChange={form.onOpenChange}>
      {open && <AddKeyForm {...form} />}
    </Dialog>
  );
}
