import React, { useState, useEffect } from "react";
import { Save, Plus, X } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { UiLoadingSpinner } from "@/components/ui/ui-loading-spinner";
import { DeprecationBanner } from "@/components/DeprecationBanner";
import { toast } from "@/lib/toast";
import {
  getGeneralSettingsCall,
  updateConfigFieldSetting,
  deleteConfigFieldSetting,
  fetchMCPClientIp,
} from "@/components/networking";

interface MCPNetworkSettingsProps {
  accessToken: string | null;
}

/**
 * Given an IP like "203.0.113.45", return "203.0.113.0/24".
 */
function ipToSlash24(ip: string): string {
  const parts = ip.split(".");
  if (parts.length !== 4) return ip + "/32";
  return `${parts[0]}.${parts[1]}.${parts[2]}.0/24`;
}

export interface AllowedClient {
  readonly alias: string;
  readonly value: string;
}

interface AllowedClientRow extends AllowedClient {
  readonly key: string;
}

const isAllowedClient = (entry: unknown): entry is AllowedClient => {
  if (typeof entry !== "object" || entry === null) return false;
  const { alias, value } = entry as Partial<Record<keyof AllowedClient, unknown>>;
  return typeof alias === "string" && typeof value === "string";
};

type StoredAllowlist =
  | { readonly kind: "absent" }
  | { readonly kind: "clients"; readonly clients: AllowedClient[] }
  | { readonly kind: "malformed" };

const ABSENT: StoredAllowlist = { kind: "absent" };

const parseStoredClients = (fieldValue: unknown): StoredAllowlist => {
  if (fieldValue === null || fieldValue === undefined) return ABSENT;
  if (Array.isArray(fieldValue) && fieldValue.every(isAllowedClient)) {
    return { kind: "clients", clients: fieldValue.map(({ alias, value }) => ({ alias, value })) };
  }
  return { kind: "malformed" };
};

let nextRowKey = 0;
const newRow = (client: AllowedClient = { alias: "", value: "" }): AllowedClientRow => ({
  ...client,
  key: `client-${nextRowKey++}`,
});

const trimClient = ({ alias, value }: AllowedClient): AllowedClient => ({ alias: alias.trim(), value: value.trim() });

const isBlank = ({ alias, value }: AllowedClient) => alias === "" && value === "";
const isIncomplete = ({ alias, value }: AllowedClient) => alias === "" || value === "";

const sameList = (a: string[], b: string[]) => a.length === b.length && a.every((value, i) => value === b[i]);

const sameClients = (a: AllowedClient[], b: AllowedClient[]) =>
  a.length === b.length && a.every((client, i) => client.alias === b[i].alias && client.value === b[i].value);

const unchangedSinceLoad = (value: string[], stored: string[] | null) =>
  stored === null ? value.length === 0 : value.length > 0 && sameList(value, stored);

const clientsUnchangedSinceLoad = (value: AllowedClient[], stored: StoredAllowlist) => {
  switch (stored.kind) {
    case "absent":
      return value.length === 0;
    case "clients":
      return value.length > 0 && sameClients(value, stored.clients);
    case "malformed":
      return false;
  }
};

const headerUnchangedSinceLoad = (value: string, stored: string | null) =>
  stored === null ? value === "" : value !== "" && value === stored;

const MCPNetworkSettings: React.FC<MCPNetworkSettingsProps> = ({ accessToken }) => {
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [privateRanges, setPrivateRanges] = useState<string[]>([]);
  const [allowedClients, setAllowedClients] = useState<AllowedClientRow[]>([]);
  const [clientIdHeader, setClientIdHeader] = useState("");
  const [storedRanges, setStoredRanges] = useState<string[] | null>(null);
  const [storedClients, setStoredClients] = useState<StoredAllowlist>(ABSENT);
  const [storedClientIdHeader, setStoredClientIdHeader] = useState<string | null>(null);
  const [currentIp, setCurrentIp] = useState<string | null>(null);
  const [rangeDraft, setRangeDraft] = useState("");

  useEffect(() => {
    loadSettings();
    detectCurrentIp();
  }, [accessToken]);

  const loadSettings = async () => {
    if (!accessToken) return;
    setLoading(true);
    try {
      const settings = await getGeneralSettingsCall(accessToken);
      for (const field of settings) {
        if (field.field_name === "mcp_internal_ip_ranges" && Array.isArray(field.field_value)) {
          setPrivateRanges(field.field_value);
          setStoredRanges(field.field_value);
        }
        if (field.field_name === "mcp_allowed_clients") {
          const stored = parseStoredClients(field.field_value);
          setAllowedClients(stored.kind === "clients" ? stored.clients.map(newRow) : []);
          setStoredClients(stored);
        }
        if (field.field_name === "mcp_client_id_header" && typeof field.field_value === "string") {
          setClientIdHeader(field.field_value);
          setStoredClientIdHeader(field.field_value);
        }
      }
    } catch (error) {
      console.error("Failed to load MCP network settings:", error);
    } finally {
      setLoading(false);
    }
  };

  const detectCurrentIp = async () => {
    if (!accessToken) return;
    const ip = await fetchMCPClientIp(accessToken);
    if (ip) {
      setCurrentIp(ip);
    }
  };

  const persistRanges = async (token: string) => {
    if (unchangedSinceLoad(privateRanges, storedRanges)) return;
    if (privateRanges.length > 0) {
      await updateConfigFieldSetting(token, "mcp_internal_ip_ranges", privateRanges);
      setStoredRanges(privateRanges);
      return;
    }
    await deleteConfigFieldSetting(token, "mcp_internal_ip_ranges");
    setStoredRanges(null);
  };

  const persistAllowedClients = async (token: string) => {
    const clients = allowedClients.map(trimClient).filter((client) => !isBlank(client));
    if (clients.some(isIncomplete)) {
      throw new Error("Every allowed client needs both an alias and a value");
    }
    if (clientsUnchangedSinceLoad(clients, storedClients)) return;
    if (clients.length > 0) {
      await updateConfigFieldSetting(token, "mcp_allowed_clients", clients);
      setStoredClients({ kind: "clients", clients });
      return;
    }
    await deleteConfigFieldSetting(token, "mcp_allowed_clients");
    setStoredClients(ABSENT);
  };

  const persistClientIdHeader = async (token: string) => {
    const value = clientIdHeader.trim();
    if (headerUnchangedSinceLoad(value, storedClientIdHeader)) return;
    if (value !== "") {
      await updateConfigFieldSetting(token, "mcp_client_id_header", value);
      setStoredClientIdHeader(value);
      return;
    }
    await deleteConfigFieldSetting(token, "mcp_client_id_header");
    setStoredClientIdHeader(null);
  };

  const handleSave = async () => {
    if (!accessToken) return;
    setSaving(true);
    const [rangeResult] = await Promise.allSettled([persistRanges(accessToken)]);
    const [clientResult] = await Promise.allSettled([persistAllowedClients(accessToken)]);
    const [headerResult] = await Promise.allSettled([persistClientIdHeader(accessToken)]);
    setSaving(false);
    const failures = [rangeResult, clientResult, headerResult].filter(
      (result): result is PromiseRejectedResult => result.status === "rejected",
    );
    if (failures.length === 0) {
      toast.success("MCP network settings saved");
      return;
    }
    failures.forEach((failure) => toast.fromError(failure.reason));
  };

  const addSuggestedRange = (range: string) => {
    if (!privateRanges.includes(range)) {
      setPrivateRanges([...privateRanges, range]);
    }
  };

  // Commas separate entries, matching the old tokenised input.
  const splitDraft = (draft: string, existing: string[]) =>
    draft
      .split(",")
      .map((r) => r.trim())
      .filter((r) => r !== "" && !existing.includes(r));

  const commitDraft = () => {
    const added = splitDraft(rangeDraft, privateRanges);
    if (added.length > 0) {
      setPrivateRanges([...privateRanges, ...added]);
    }
    setRangeDraft("");
  };

  const updateClient = (key: string, patch: Partial<AllowedClient>) =>
    setAllowedClients(allowedClients.map((row) => (row.key === key ? { ...row, ...patch } : row)));

  const removeClient = (key: string) => setAllowedClients(allowedClients.filter((row) => row.key !== key));

  if (loading) {
    return (
      <div className="flex justify-center py-12">
        <UiLoadingSpinner className="size-6 text-muted-foreground" />
      </div>
    );
  }

  const suggestedRange = currentIp ? ipToSlash24(currentIp) : null;
  const storedAllowlistIsMalformed = storedClients.kind === "malformed";
  const storedAllowlistIsEmpty = storedClients.kind === "clients" && storedClients.clients.length === 0;

  return (
    <div className="space-y-6 p-4">
      <DeprecationBanner featureName="MCP Network Settings and the internal-network-only flag" />
      <div>
        <p className="text-lg font-semibold">Private IP Ranges</p>
        <p className="mt-1 text-sm text-muted-foreground">
          Define which IP ranges are part of your private network. Callers from these IPs can see all MCP servers.
          Callers from any other IP can only see servers marked &quot;Available on Public Internet&quot;.
        </p>
      </div>

      <Card className="p-6">
        {currentIp && (
          <div className="mb-4 rounded-lg bg-muted p-3">
            <p className="text-sm">
              Your current IP: <span className="font-mono font-medium">{currentIp}</span>
            </p>
            {suggestedRange && !privateRanges.includes(suggestedRange) && (
              <div className="mt-1 flex items-center gap-2">
                <p className="text-sm">Suggested range: </p>
                <Button
                  variant="outline"
                  size="sm"
                  className="font-mono"
                  onClick={() => addSuggestedRange(suggestedRange)}
                >
                  <Plus />
                  {suggestedRange}
                </Button>
              </div>
            )}
          </div>
        )}

        <div className="mb-2 flex items-center">
          <p className="text-sm font-medium">Your Private Network Ranges</p>
        </div>
        {privateRanges.length > 0 && (
          <div className="mb-2 flex flex-wrap gap-1.5">
            {privateRanges.map((range) => (
              <Badge key={range} variant="secondary" className="font-mono">
                {range}
                <button
                  type="button"
                  aria-label={`Remove ${range}`}
                  onClick={() => setPrivateRanges(privateRanges.filter((r) => r !== range))}
                  className="ml-1 cursor-pointer"
                >
                  <X className="size-3" />
                </button>
              </Badge>
            ))}
          </div>
        )}
        <Input
          value={rangeDraft}
          placeholder="Leave empty to use defaults: 10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16, 127.0.0.0/8"
          onChange={(e) => setRangeDraft(e.target.value)}
          onBlur={commitDraft}
          onKeyDown={(e) => {
            if (e.key === "Enter" || e.key === ",") {
              e.preventDefault();
              commitDraft();
            }
          }}
        />
        <p className="mt-2 text-xs text-muted-foreground">
          Enter CIDR ranges (e.g., 10.0.0.0/8). When empty, standard private IP ranges are used.
        </p>
      </Card>

      <div>
        <p className="text-lg font-semibold">Allowed Client Applications</p>
        <p className="mt-1 text-sm text-muted-foreground">
          Only the MCP client applications listed here can use the gateway. Leave empty to allow every client. A client
          that authenticates with a JWT is identified by the claim named in litellm_jwtauth.mcp_client_id_jwt_field in
          your proxy config (for example azp or client_id), which your identity provider asserts and the client cannot
          change. Any other client is identified by the request header configured below, if you enable one.
        </p>
      </div>

      <Card className="p-6">
        <div className="mb-2 flex items-center">
          <p className="text-sm font-medium">Allowed Clients</p>
        </div>
        {storedAllowlistIsMalformed && (
          <p className="mb-2 text-sm text-destructive">
            The stored allowlist is not a list of alias and value pairs, so every client is denied. Add the clients you
            want and save to replace it, or save with the list empty to remove it and allow every client again.
          </p>
        )}
        {storedAllowlistIsEmpty && (
          <p className="mb-2 text-sm text-destructive">
            An empty allowlist is currently stored, so every client is denied. Save with the list empty to remove it and
            allow every client again.
          </p>
        )}
        {allowedClients.length > 0 && (
          <div className="mb-2 grid grid-cols-[1fr_1fr_auto] items-center gap-2">
            <p className="text-xs text-muted-foreground">Alias</p>
            <p className="text-xs text-muted-foreground">Value</p>
            <span />
            {allowedClients.map((row, index) => (
              <React.Fragment key={row.key}>
                <Input
                  aria-label={`Client ${index + 1} alias`}
                  value={row.alias}
                  placeholder="e.g. Coding CLI"
                  onChange={(e) => updateClient(row.key, { alias: e.target.value })}
                />
                <Input
                  aria-label={`Client ${index + 1} value`}
                  value={row.value}
                  placeholder="e.g. 0oa1b2c3d4e5f6g7h8i9"
                  className="font-mono"
                  onChange={(e) => updateClient(row.key, { value: e.target.value })}
                />
                <Button
                  type="button"
                  variant="ghost"
                  size="icon"
                  aria-label={`Remove client ${row.alias.trim() || index + 1}`}
                  onClick={() => removeClient(row.key)}
                >
                  <X className="size-4" />
                </Button>
              </React.Fragment>
            ))}
          </div>
        )}
        <Button
          type="button"
          variant="outline"
          size="sm"
          onClick={() => setAllowedClients([...allowedClients, newRow()])}
        >
          <Plus />
          Add client
        </Button>
        <p className="mt-2 text-xs text-muted-foreground">
          The alias is the name shown here and in gateway logs. The value is the exact JWT claim or header value that
          identifies the client, such as the OAuth client ID your identity provider issues. Leave the list empty to
          allow every client. Every MCP request from an unlisted client, or from one with no resolvable identity, gets a
          403.
        </p>

        <div className="mt-6 mb-2 flex items-center">
          <p className="text-sm font-medium">Client Identity Header (less secure)</p>
        </div>
        <Input
          aria-label="Client identity header"
          value={clientIdHeader}
          placeholder="Leave empty to identify clients by JWT only, e.g. x-mcp-client"
          onChange={(e) => setClientIdHeader(e.target.value)}
        />
        <p className="mt-2 text-xs text-muted-foreground">
          Optional header whose value names the client for callers without a JWT identity. Clients pick this value
          themselves, so it is a policy control rather than a security boundary. Without it, callers that do not carry
          the JWT claim are rejected while the allowlist is set.
        </p>
      </Card>

      <div className="flex justify-end">
        <Button onClick={handleSave} disabled={saving}>
          <Save />
          Save
        </Button>
      </div>
    </div>
  );
};

export default MCPNetworkSettings;
