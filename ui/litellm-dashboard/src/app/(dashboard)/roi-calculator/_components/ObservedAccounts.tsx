"use client";

import { useEffect, useState } from "react";
import { z } from "zod";
import { apiClient } from "@/components/networking";
import { extractProxyErrorMessage } from "@/lib/http/client";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "@/components/ui/dialog";
import { accountLogins, type ObservedPerson } from "./observedData";

const connectionIdentityFields = {
  id: z.string(),
  source_provider: z.enum(["github", "gitlab"]),
  api_url: z.string(),
  identity_map: z.record(z.string(), z.string()),
  unmatched_logins: z.array(z.string()),
};
const identitiesFields = {
  gateway_emails: z.array(z.string()),
  identity_map: z.record(z.string(), z.string()),
  unmatched_logins: z.array(z.string()),
  connections: z.array(z.object(connectionIdentityFields)).optional(),
};
const identitiesSchema = z.object(identitiesFields);

function matches(identities: z.infer<typeof identitiesSchema>, email: string, people: ObservedPerson[]) {
  const person = people.find((entry) => entry.email === email);
  return Object.fromEntries(
    (identities.connections ?? []).map((entry) => [
      entry.id,
      [
        ...new Set([
          ...Object.entries(entry.identity_map)
            .filter(([, address]) => address === email)
            .map(([login]) => login),
          ...(person?.accounts ?? [])
            .filter((account) => account.connection_id === entry.id)
            .map((account) => account.login),
        ]),
      ].join(", "),
    ]),
  );
}

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
  const [linked, setLinked] = useState<Record<string, string>>({});
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  useEffect(() => {
    const controller = new AbortController();
    apiClient
      .get<unknown>("/roi-calculator/observed/identities", { accessToken, signal: controller.signal })
      .then((data) => {
        if (!controller.signal.aborted) {
          const parsed = identitiesSchema.parse(data);
          setIdentities(parsed);
          setLinked(matches(parsed, initialEmail, people));
        }
      })
      .catch((reason: unknown) => {
        if (!controller.signal.aborted) setError(extractProxyErrorMessage(reason));
      });
    return () => controller.abort();
  }, [accessToken, initialEmail, people]);
  function selectEmail(value: string) {
    setEmail(value);
    if (identities) setLinked(matches(identities, value, people));
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
        body: {
          email: email.trim().toLowerCase(),
          ...(identities?.connections?.length
            ? {
                accounts: identities.connections.flatMap((entry) =>
                  accountLogins(linked[entry.id] ?? "").map((login) => ({ connection_id: entry.id, login })),
                ),
              }
            : { logins: accountLogins(logins) }),
        },
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
              disabled={!identities}
              list="roi-internal-emails"
              value={email}
              placeholder="Choose an internal user"
              onChange={(event) => selectEmail(event.target.value)}
            />
            <datalist id="roi-internal-emails">
              {identities?.gateway_emails.map((address) => <option key={address} value={address} />)}
            </datalist>
          </div>
          {identities?.connections?.length ? (
            identities.connections.map((entry) => (
              <div key={entry.id} className="space-y-2">
                <label htmlFor={`identity-${entry.id}`} className="text-sm font-medium">
                  {entry.source_provider === "github" ? "GitHub" : "GitLab"} usernames
                  <span className="ml-2 text-xs font-normal text-muted-foreground">{new URL(entry.api_url).host}</span>
                </label>
                <Input
                  id={`identity-${entry.id}`}
                  value={linked[entry.id] ?? ""}
                  onChange={(event) => setLinked({ ...linked, [entry.id]: event.target.value })}
                  placeholder="current-account, old-account"
                />
                {entry.unmatched_logins.length > 0 && (
                  <details className="text-xs text-muted-foreground">
                    <summary className="cursor-pointer">{entry.unmatched_logins.length} unmatched accounts</summary>
                    <div className="mt-2 max-h-32 overflow-auto flex flex-wrap gap-1">
                      {entry.unmatched_logins.map((login) => (
                        <Button
                          key={login}
                          size="sm"
                          variant="outline"
                          onClick={() =>
                            setLinked({
                              ...linked,
                              [entry.id]: [...new Set([...accountLogins(linked[entry.id] ?? ""), login])].join(", "),
                            })
                          }
                        >
                          {login}
                        </Button>
                      ))}
                    </div>
                  </details>
                )}
              </div>
            ))
          ) : (
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
            </div>
          )}
          <p className="text-xs text-muted-foreground">
            Separate accounts with commas. Their merged changes are combined, and gateway spend is counted once
          </p>
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
