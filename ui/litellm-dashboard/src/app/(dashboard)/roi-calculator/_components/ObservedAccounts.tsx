"use client";

import { useEffect, useState } from "react";
import { z } from "zod";
import { apiClient } from "@/components/networking";
import { extractProxyErrorMessage } from "@/lib/http/client";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "@/components/ui/dialog";
import { accountLogins, type ObservedPerson } from "./observedData";

const identitiesSchema = z.object({
  gateway_emails: z.array(z.string()),
  identity_map: z.record(z.string(), z.string()),
  unmatched_logins: z.array(z.string()),
});

export default function ObservedAccounts({
  accessToken,
  people,
  initialEmail = "",
  onClose,
  onSaved,
}: {
  accessToken: string;
  people: ObservedPerson[];
  initialEmail?: string;
  onClose: () => void;
  onSaved: () => void;
}) {
  const [identities, setIdentities] = useState<z.infer<typeof identitiesSchema> | null>(null);
  const [email, setEmail] = useState(initialEmail);
  const [logins, setLogins] = useState(people.find((person) => person.email === initialEmail)?.logins.join(", ") ?? "");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  useEffect(() => {
    const controller = new AbortController();
    apiClient
      .get<unknown>("/roi-calculator/observed/identities", { accessToken, signal: controller.signal })
      .then((data) => {
        if (!controller.signal.aborted) setIdentities(identitiesSchema.parse(data));
      })
      .catch((reason: unknown) => {
        if (!controller.signal.aborted) setError(extractProxyErrorMessage(reason));
      });
    return () => controller.abort();
  }, [accessToken]);
  function selectEmail(value: string) {
    setEmail(value);
    const automatic = people.find((person) => person.email === value)?.logins ?? [];
    const manual = Object.entries(identities?.identity_map ?? {})
      .filter(([, address]) => address === value)
      .map(([login]) => login);
    setLogins([...new Set([...automatic, ...manual])].join(", "));
  }
  async function save() {
    setSaving(true);
    setError("");
    try {
      await apiClient.put<unknown>("/roi-calculator/observed/identities", {
        accessToken,
        body: { email: email.trim().toLowerCase(), logins: accountLogins(logins) },
      });
      onSaved();
      onClose();
    } catch (reason) {
      setError(extractProxyErrorMessage(reason));
    } finally {
      setSaving(false);
    }
  }
  return (
    <Dialog
      open
      onOpenChange={(open) => {
        if (!open) onClose();
      }}
    >
      <DialogContent className="sm:max-w-lg">
        <DialogHeader>
          <DialogTitle>Link accounts</DialogTitle>
          <DialogDescription>Match one internal user to all their source accounts</DialogDescription>
        </DialogHeader>
        <div className="space-y-4 py-2">
          <div className="space-y-2">
            <label htmlFor="identity-email" className="text-sm font-medium">
              Internal email
            </label>
            <Input
              id="identity-email"
              list="roi-internal-emails"
              value={email}
              placeholder="Choose an internal user"
              onChange={(event) => selectEmail(event.target.value)}
            />
            <datalist id="roi-internal-emails">
              {identities?.gateway_emails.map((address) => <option key={address} value={address} />)}
            </datalist>
          </div>
          <div className="space-y-2">
            <label htmlFor="identity-logins" className="text-sm font-medium">
              Source usernames
            </label>
            <Input
              id="identity-logins"
              value={logins}
              onChange={(event) => setLogins(event.target.value)}
              placeholder="current-account, old-account"
            />
            <p className="text-xs text-muted-foreground">
              Separate accounts with commas. All their merged changes count toward this person
            </p>
          </div>
          {identities && identities.unmatched_logins.length > 0 && (
            <details className="text-xs text-muted-foreground">
              <summary className="cursor-pointer">{identities.unmatched_logins.length} unmatched accounts</summary>
              <div className="mt-2 max-h-32 overflow-auto flex flex-wrap gap-1">
                {identities.unmatched_logins.map((login) => (
                  <Button
                    key={login}
                    size="sm"
                    variant="outline"
                    onClick={() => setLogins([...new Set([...accountLogins(logins), login])].join(", "))}
                  >
                    {login}
                  </Button>
                ))}
              </div>
            </details>
          )}
          {error && (
            <p role="alert" className="text-sm text-destructive">
              {error}
            </p>
          )}
          <Button className="w-full" disabled={!identities || !email.trim() || saving} onClick={save}>
            {saving ? "Saving…" : "Save accounts"}
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}
