"use client";

import { useState } from "react";
import Link from "next/link";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { uiHref } from "@/utils/uiHref";
import { customerLimitsOnKey, writeCustomerLimits, type PortalKey } from "./customerKeys";

interface Props {
  open: boolean;
  onOpenChange: (o: boolean) => void;
  accessToken: string;
  customerId: string;
  customerLabel: string;
  keys: PortalKey[];
  excludeTokens: ReadonlySet<string>;
  editKey: PortalKey | null;
  onSaved: () => void;
}

const digits = (s: string) => s.replace(/[^0-9]/g, "");
const num = (s: string) => (s ? Number(s) : null);
const asText = (n: number | null) => (n === null ? "" : String(n));

const successMessage = (editKey: PortalKey | null, alias: string, customerLabel: string, cleared: boolean) => {
  if (cleared) return `Removed ${alias} limits for ${customerLabel}`;
  if (editKey) return `Rate limits updated for ${customerLabel}`;
  return `Added ${alias} to ${customerLabel}`;
};

type FormProps = Omit<Props, "open">;

function LimitForm({ onOpenChange, accessToken, customerId, customerLabel, keys, excludeTokens, editKey, onSaved }: FormProps) {
  const available = editKey ? [editKey] : keys.filter((k) => !excludeTokens.has(k.token));
  const initial = editKey ?? available[0] ?? null;
  const initialLimits = initial ? customerLimitsOnKey(initial, customerId) : { tpm: null, rpm: null };
  const [token, setToken] = useState(initial?.token ?? "");
  const [tpm, setTpm] = useState(asText(initialLimits.tpm));
  const [rpm, setRpm] = useState(asText(initialLimits.rpm));
  const [saving, setSaving] = useState(false);
  const selected = keys.find((k) => k.token === token) ?? null;

  const pickKey = (next: string | null) => {
    const k = keys.find((x) => x.token === next) ?? null;
    const cur = k ? customerLimitsOnKey(k, customerId) : { tpm: null, rpm: null };
    setToken(next ?? "");
    setTpm(asText(cur.tpm));
    setRpm(asText(cur.rpm));
  };

  const save = async (clear = false) => {
    if (!selected) return;
    setSaving(true);
    try {
      await writeCustomerLimits(accessToken, selected, customerId, clear ? { tpm: null, rpm: null } : { tpm: num(tpm), rpm: num(rpm) });
      toast.success(successMessage(editKey, selected.alias, customerLabel, clear));
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
        <DialogTitle>{editKey ? `Edit limits for ${editKey.alias}` : `Add key to ${customerLabel}`}</DialogTitle>
        <DialogDescription>Limits apply to this customer on this key only. Other customers on the same key are unaffected.</DialogDescription>
      </DialogHeader>
      <div className="space-y-4">
        {!editKey && (
          <div className="space-y-1.5">
            <Label htmlFor="ak-key">Key</Label>
            <Select value={token} onValueChange={pickKey}>
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
        )}
        <div className="grid gap-4 sm:grid-cols-2">
          <div className="space-y-1.5">
            <Label htmlFor="ak-tpm">TPM limit for this customer</Label>
            <Input id="ak-tpm" inputMode="numeric" value={tpm} onChange={(e) => setTpm(digits(e.target.value))} placeholder={selected?.tpmLimit ? `Key limit ${selected.tpmLimit.toLocaleString()}` : "No limit"} />
          </div>
          <div className="space-y-1.5">
            <Label htmlFor="ak-rpm">RPM limit for this customer</Label>
            <Input id="ak-rpm" inputMode="numeric" value={rpm} onChange={(e) => setRpm(digits(e.target.value))} placeholder={selected?.rpmLimit ? `Key limit ${selected.rpmLimit.toLocaleString()}` : "No limit"} />
          </div>
        </div>
      </div>
      <DialogFooter>
        {editKey && <Button type="button" variant="ghost" className="mr-auto text-destructive" disabled={saving} onClick={() => save(true)}>Remove key</Button>}
        <Button type="button" variant="outline" onClick={() => onOpenChange(false)}>Cancel</Button>
        <Button type="button" disabled={!selected || saving} onClick={() => save(false)}>{editKey ? "Save changes" : "Add key"}</Button>
      </DialogFooter>
    </DialogContent>
  );
}

export default function KeyLimitDialog({ open, ...form }: Props) {
  return (
    <Dialog open={open} onOpenChange={form.onOpenChange}>
      {open && <LimitForm key={form.editKey?.token ?? "new"} {...form} />}
    </Dialog>
  );
}
