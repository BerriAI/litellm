import React, { useState, useEffect, useId } from "react";
import { useController } from "react-hook-form";
import type { ParseKeys } from "i18next";
import { useTranslation } from "react-i18next";
import { Save, Plus, X } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
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
import { Switch } from "@/components/ui/switch";
import { UiLoadingSpinner } from "@/components/ui/ui-loading-spinner";
import { useZodForm } from "@/lib/forms/useZodForm";
import { toast } from "@/lib/toast";
import {
  getGeneralSettingsCall,
  updateConfigFieldSetting,
  deleteConfigFieldSetting,
  fetchMCPClientIp,
} from "@/components/networking";
import {
  ABSENT,
  clientsUnchangedSinceLoad,
  headerUnchangedSinceLoad,
  isIncomplete,
  parseDisabledCapabilities,
  parseStoredClients,
  toAllowedClients,
  trimClient,
  unchangedSinceLoad,
  type AllowedClient,
  type StoredAllowlist,
} from "./MCPNetworkSettings/mapper";
import {
  defaultNetworkSettings,
  mcpNetworkSettingsSchema,
  type ClientDraft,
  type DisableableCapability,
} from "./MCPNetworkSettings/schema";

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

interface AllowedClientRow extends AllowedClient {
  readonly key: string;
}

let nextRowKey = 0;
const newRow = (client: AllowedClient): AllowedClientRow => ({ ...client, key: `client-${nextRowKey++}` });

interface AllowedClientDialogProps {
  readonly draft: ClientDraft | null;
  readonly onChange: (draft: ClientDraft) => void;
  readonly onCommit: () => void;
  readonly onRemove: () => void;
  readonly onClose: () => void;
}

const AllowedClientDialog: React.FC<AllowedClientDialogProps> = ({ draft, onChange, onCommit, onRemove, onClose }) => {
  const { t } = useTranslation();
  const aliasId = useId();
  const valueId = useId();
  if (draft === null) return null;
  return (
    <Dialog open onOpenChange={(open) => !open && onClose()}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>
            {draft.key === null
              ? t("mcpServers.networkSettings.clientDialog.addTitle")
              : t("mcpServers.networkSettings.clientDialog.editTitle")}
          </DialogTitle>
          <DialogDescription>{t("mcpServers.networkSettings.clientDialog.description")}</DialogDescription>
        </DialogHeader>
        <div className="grid gap-4">
          <div className="grid gap-2">
            <Label htmlFor={aliasId}>{t("mcpServers.networkSettings.clientDialog.aliasLabel")}</Label>
            <Input
              id={aliasId}
              value={draft.alias}
              placeholder={t("mcpServers.networkSettings.clientDialog.aliasPlaceholder")}
              onChange={(e) => onChange({ ...draft, alias: e.target.value })}
            />
          </div>
          <div className="grid gap-2">
            <Label htmlFor={valueId}>{t("mcpServers.networkSettings.clientDialog.valueLabel")}</Label>
            <Input
              id={valueId}
              value={draft.value}
              placeholder={t("mcpServers.networkSettings.clientDialog.valuePlaceholder")}
              className="font-mono"
              onChange={(e) => onChange({ ...draft, value: e.target.value })}
            />
          </div>
        </div>
        <DialogFooter>
          {draft.key !== null && (
            <Button type="button" variant="destructive" className="sm:mr-auto" onClick={onRemove}>
              {t("mcpServers.networkSettings.clientDialog.remove")}
            </Button>
          )}
          <Button type="button" variant="outline" onClick={onClose}>
            {t("mcpServers.networkSettings.clientDialog.cancel")}
          </Button>
          <Button type="button" disabled={isIncomplete(trimClient(draft))} onClick={onCommit}>
            {draft.key === null
              ? t("mcpServers.networkSettings.clientDialog.add")
              : t("mcpServers.networkSettings.clientDialog.done")}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};

const CAPABILITY_TOGGLES: { capability: DisableableCapability; labelKey: ParseKeys; descriptionKey: ParseKeys }[] = [
  {
    capability: "prompts",
    labelKey: "mcpServers.networkSettings.capabilities.prompts.label",
    descriptionKey: "mcpServers.networkSettings.capabilities.prompts.description",
  },
  {
    capability: "resources",
    labelKey: "mcpServers.networkSettings.capabilities.resources.label",
    descriptionKey: "mcpServers.networkSettings.capabilities.resources.description",
  },
];

const MCPNetworkSettings: React.FC<MCPNetworkSettingsProps> = ({ accessToken }) => {
  const { t } = useTranslation();
  const { control, getValues } = useZodForm(mcpNetworkSettingsSchema, { defaultValues: defaultNetworkSettings });
  const {
    field: { value: privateRanges, onChange: setPrivateRanges },
  } = useController({ control, name: "privateRanges" });
  const {
    field: { value: allowedClients, onChange: setAllowedClients },
  } = useController({ control, name: "allowedClients" });
  const {
    field: { value: clientIdHeader, onChange: setClientIdHeader },
  } = useController({ control, name: "clientIdHeader" });
  const {
    field: { value: disabledCapabilities, onChange: setDisabledCapabilities },
  } = useController({ control, name: "disabledCapabilities" });
  const {
    field: { value: rangeDraft, onChange: setRangeDraft },
  } = useController({ control, name: "rangeDraft" });
  const {
    field: { value: clientDraft, onChange: setClientDraft },
  } = useController({ control, name: "clientDraft" });
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [storedRanges, setStoredRanges] = useState<string[] | null>(null);
  const [storedClients, setStoredClients] = useState<StoredAllowlist>(ABSENT);
  const [storedClientIdHeader, setStoredClientIdHeader] = useState<string | null>(null);
  const [storedDisabledCapabilities, setStoredDisabledCapabilities] = useState<DisableableCapability[] | null>(null);
  const [currentIp, setCurrentIp] = useState<string | null>(null);
  const capabilityToggleId = useId();

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
        if (field.field_name === "mcp_disabled_capabilities") {
          const stored = parseDisabledCapabilities(field.field_value);
          setDisabledCapabilities(stored);
          setStoredDisabledCapabilities(stored.length > 0 ? stored : null);
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

  const persistRanges = async (token: string, privateRanges: string[]) => {
    if (unchangedSinceLoad(privateRanges, storedRanges)) return;
    if (privateRanges.length > 0) {
      await updateConfigFieldSetting(token, "mcp_internal_ip_ranges", privateRanges);
      setStoredRanges(privateRanges);
      return;
    }
    await deleteConfigFieldSetting(token, "mcp_internal_ip_ranges");
    setStoredRanges(null);
  };

  const persistAllowedClients = async (token: string, allowedClients: AllowedClientRow[]) => {
    const clients = toAllowedClients(allowedClients);
    if (clientsUnchangedSinceLoad(clients, storedClients)) return;
    if (clients.length > 0) {
      await updateConfigFieldSetting(token, "mcp_allowed_clients", clients);
      setStoredClients({ kind: "clients", clients });
      return;
    }
    await deleteConfigFieldSetting(token, "mcp_allowed_clients");
    setStoredClients(ABSENT);
  };

  const persistClientIdHeader = async (token: string, clientIdHeader: string) => {
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

  const persistDisabledCapabilities = async (token: string, disabledCapabilities: DisableableCapability[]) => {
    if (unchangedSinceLoad(disabledCapabilities, storedDisabledCapabilities)) return;
    if (disabledCapabilities.length > 0) {
      await updateConfigFieldSetting(token, "mcp_disabled_capabilities", disabledCapabilities);
      setStoredDisabledCapabilities(disabledCapabilities);
      return;
    }
    await deleteConfigFieldSetting(token, "mcp_disabled_capabilities");
    setStoredDisabledCapabilities(null);
  };

  const setCapabilityServed = (capability: DisableableCapability, served: boolean) => {
    setDisabledCapabilities(
      served
        ? disabledCapabilities.filter((name) => name !== capability)
        : [...disabledCapabilities.filter((name) => name !== capability), capability],
    );
  };

  const handleSave = async () => {
    if (!accessToken) return;
    const draft = mcpNetworkSettingsSchema.safeParse(getValues());
    if (!draft.success) return;
    setSaving(true);
    const [rangeResult] = await Promise.allSettled([persistRanges(accessToken, draft.data.privateRanges)]);
    const [clientResult] = await Promise.allSettled([persistAllowedClients(accessToken, draft.data.allowedClients)]);
    const [headerResult] = await Promise.allSettled([persistClientIdHeader(accessToken, draft.data.clientIdHeader)]);
    const [capabilityResult] = await Promise.allSettled([
      persistDisabledCapabilities(accessToken, draft.data.disabledCapabilities),
    ]);
    setSaving(false);
    const failures = [rangeResult, clientResult, headerResult, capabilityResult].filter(
      (result): result is PromiseRejectedResult => result.status === "rejected",
    );
    if (failures.length === 0) {
      toast.success(t("mcpServers.networkSettings.saved"));
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

  const commitClientDraft = () => {
    if (clientDraft === null) return;
    const client = trimClient(clientDraft);
    setAllowedClients(
      clientDraft.key === null
        ? [...allowedClients, newRow(client)]
        : allowedClients.map((row) => (row.key === clientDraft.key ? { ...row, ...client } : row)),
    );
    setClientDraft(null);
  };

  const removeDraftedClient = () => {
    if (clientDraft === null) return;
    setAllowedClients(allowedClients.filter((row) => row.key !== clientDraft.key));
    setClientDraft(null);
  };

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
      <div>
        <p className="text-lg font-semibold">{t("mcpServers.networkSettings.privateRanges.title")}</p>
        <p className="mt-1 text-sm text-muted-foreground">
          {t("mcpServers.networkSettings.privateRanges.description")}
        </p>
      </div>

      <Card className="p-6">
        {currentIp && (
          <div className="mb-4 rounded-lg bg-muted p-3">
            <p className="text-sm">
              {t("mcpServers.networkSettings.privateRanges.currentIp")}{" "}
              <span className="font-mono font-medium">{currentIp}</span>
            </p>
            {suggestedRange && !privateRanges.includes(suggestedRange) && (
              <div className="mt-1 flex items-center gap-2">
                <p className="text-sm">{t("mcpServers.networkSettings.privateRanges.suggestedRange")} </p>
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
          <p className="text-sm font-medium">{t("mcpServers.networkSettings.privateRanges.listTitle")}</p>
        </div>
        {privateRanges.length > 0 && (
          <div className="mb-2 flex flex-wrap gap-1.5">
            {privateRanges.map((range) => (
              <Badge key={range} variant="secondary" className="font-mono">
                {range}
                <button
                  type="button"
                  aria-label={t("mcpServers.networkSettings.privateRanges.removeAriaLabel", { range })}
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
          placeholder={t("mcpServers.networkSettings.privateRanges.placeholder")}
          onChange={(e) => setRangeDraft(e.target.value)}
          onBlur={commitDraft}
          onKeyDown={(e) => {
            if (e.key === "Enter" || e.key === ",") {
              e.preventDefault();
              commitDraft();
            }
          }}
        />
        <p className="mt-2 text-xs text-muted-foreground">{t("mcpServers.networkSettings.privateRanges.hint")}</p>
      </Card>

      <div>
        <p className="text-lg font-semibold">{t("mcpServers.networkSettings.allowedClients.title")}</p>
        <p className="mt-1 text-sm text-muted-foreground">
          {t("mcpServers.networkSettings.allowedClients.description")}
        </p>
      </div>

      <Card className="p-6">
        {storedAllowlistIsMalformed && (
          <p className="mb-2 text-sm text-destructive">
            {t("mcpServers.networkSettings.allowedClients.malformedWarning")}
          </p>
        )}
        {storedAllowlistIsEmpty && (
          <p className="mb-2 text-sm text-destructive">{t("mcpServers.networkSettings.allowedClients.emptyWarning")}</p>
        )}
        {allowedClients.length > 0 && (
          <div className="mb-3 grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
            {allowedClients.map((row) => (
              <button
                key={row.key}
                type="button"
                className="flex min-w-0 flex-col items-start gap-1 rounded-lg border border-border bg-background p-3 text-left hover:bg-muted focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none"
                onClick={() => setClientDraft(row)}
              >
                <span className="w-full truncate text-sm font-medium">{row.alias}</span>
                <span className="w-full truncate font-mono text-xs text-muted-foreground">{row.value}</span>
              </button>
            ))}
          </div>
        )}
        <Button
          type="button"
          variant="outline"
          size="sm"
          onClick={() => setClientDraft({ key: null, alias: "", value: "" })}
        >
          <Plus />
          {t("mcpServers.networkSettings.allowedClients.add")}
        </Button>
        <p className="mt-2 text-xs text-muted-foreground">{t("mcpServers.networkSettings.allowedClients.hint")}</p>

        <div className="mt-6 mb-2 flex items-center">
          <p className="text-sm font-medium">{t("mcpServers.networkSettings.clientIdHeader.title")}</p>
        </div>
        <Input
          aria-label={t("mcpServers.networkSettings.clientIdHeader.ariaLabel")}
          value={clientIdHeader}
          placeholder={t("mcpServers.networkSettings.clientIdHeader.placeholder")}
          onChange={(e) => setClientIdHeader(e.target.value)}
        />
        <p className="mt-2 text-xs text-muted-foreground">{t("mcpServers.networkSettings.clientIdHeader.hint")}</p>
      </Card>

      <div>
        <p className="text-lg font-semibold">{t("mcpServers.networkSettings.capabilities.title")}</p>
        <p className="mt-1 text-sm text-muted-foreground">{t("mcpServers.networkSettings.capabilities.description")}</p>
      </div>

      <Card className="space-y-4 p-6">
        {CAPABILITY_TOGGLES.map(({ capability, labelKey, descriptionKey }) => (
          <div key={capability} className="flex items-center justify-between gap-4">
            <div>
              <Label htmlFor={`${capabilityToggleId}-${capability}`} className="text-sm font-medium">
                {t(labelKey)}
              </Label>
              <p className="text-xs text-muted-foreground">{t(descriptionKey)}</p>
            </div>
            <Switch
              id={`${capabilityToggleId}-${capability}`}
              size="sm"
              checked={!disabledCapabilities.includes(capability)}
              onCheckedChange={(served) => setCapabilityServed(capability, served)}
            />
          </div>
        ))}
      </Card>

      <div className="flex justify-end">
        <Button onClick={handleSave} disabled={saving}>
          <Save />
          {t("mcpServers.networkSettings.save")}
        </Button>
      </div>

      <AllowedClientDialog
        draft={clientDraft}
        onChange={setClientDraft}
        onCommit={commitClientDraft}
        onRemove={removeDraftedClient}
        onClose={() => setClientDraft(null)}
      />
    </div>
  );
};

export default MCPNetworkSettings;
