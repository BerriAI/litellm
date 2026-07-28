"use client";

import React, { useCallback, useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { Plus, Settings2 } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { Switch } from "@/components/ui/switch";
import { createGuardrailCall, getGuardrailsList, updateGuardrailCall } from "@/components/networking";
import NotificationsManager from "@/components/molecules/notifications_manager";
import { guardrailDetailHref } from "@/app/(dashboard)/guardrails/detailNavigation";
import { isAdminRole } from "@/utils/roles";
import {
  buildCompressionGuardrailPayload,
  compressionGuardrailsOf,
  GuardrailListItem,
  GuardrailListResponse,
  isConfigDefinedGuardrail,
} from "./helpers";

interface PromptCompressionTabProps {
  accessToken: string | null;
  userRole?: string;
}

type CompressionMode = "always" | "opt_in";

const MODE_LABELS: Readonly<Record<CompressionMode, string>> = {
  always: "Always on",
  opt_in: "Opt-in",
};

const modeOf = (guardrail: GuardrailListItem): CompressionMode =>
  guardrail.litellm_params?.default_on ? "always" : "opt_in";

interface CompressionEndpointRowProps {
  guardrail: GuardrailListItem;
  canEdit: boolean;
  isPending: boolean;
  onModeChange: (guardrail: GuardrailListItem, mode: CompressionMode) => void;
  onEditSettings: (guardrail: GuardrailListItem) => void;
}

const CompressionEndpointRow: React.FC<CompressionEndpointRowProps> = ({
  guardrail,
  canEdit,
  isPending,
  onModeChange,
  onEditSettings,
}) => {
  const mode = modeOf(guardrail);
  const name = guardrail.guardrail_name ?? guardrail.guardrail_id;
  const isFromConfig = isConfigDefinedGuardrail(guardrail);
  const isEditable = canEdit && !isFromConfig;

  const handleSelect = (value: unknown) => {
    if (value !== "always" && value !== "opt_in") {
      return;
    }
    onModeChange(guardrail, value);
  };

  return (
    <li className="flex flex-wrap items-center justify-between gap-4 p-4">
      <div className="min-w-0">
        <p className="truncate text-sm font-medium text-foreground">{name}</p>
        <p className="truncate text-xs text-muted-foreground">{guardrail.litellm_params?.api_base ?? ""}</p>
        {isFromConfig && (
          <p className="mt-1 text-xs text-muted-foreground">
            Defined in the proxy config file, so it is read-only here
          </p>
        )}
      </div>

      <div className="flex items-center gap-2">
        {isEditable ? (
          <Select value={mode} onValueChange={handleSelect} disabled={isPending}>
            <SelectTrigger size="sm" className="w-[124px]" aria-label={`Compression mode for ${name}`}>
              <SelectValue>{MODE_LABELS[mode]}</SelectValue>
            </SelectTrigger>
            <SelectContent>
              {Object.entries(MODE_LABELS).map(([value, label]) => (
                <SelectItem key={value} value={value}>
                  {label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        ) : (
          <Badge variant={mode === "always" ? "secondary" : "outline"}>{MODE_LABELS[mode]}</Badge>
        )}

        {canEdit && (
          <Button variant="outline" size="sm" onClick={() => onEditSettings(guardrail)}>
            <Settings2 />
            Edit settings
          </Button>
        )}
      </div>

      {mode === "opt_in" && (
        <p className="w-full rounded-md bg-muted px-3 py-2 text-xs text-muted-foreground">
          Compression only runs when a request asks for it:{" "}
          <code className="font-mono text-foreground">&quot;guardrails&quot;: [&quot;{name}&quot;]</code> in the request
          body. Turning it on for a specific key or team instead is a LiteLLM Enterprise feature
        </p>
      )}
    </li>
  );
};

const PromptCompressionTab: React.FC<PromptCompressionTabProps> = ({ accessToken, userRole }) => {
  const router = useRouter();
  const isAdmin = userRole ? isAdminRole(userRole) : false;

  const [guardrails, setGuardrails] = useState<GuardrailListItem[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [isSaving, setIsSaving] = useState<boolean>(false);
  const [pendingModeId, setPendingModeId] = useState<string | null>(null);
  const [isAddFormOpen, setIsAddFormOpen] = useState<boolean>(false);
  const [name, setName] = useState<string>("");
  const [apiBase, setApiBase] = useState<string>("");
  const [defaultOn, setDefaultOn] = useState<boolean>(true);
  const [showFieldErrors, setShowFieldErrors] = useState<boolean>(false);

  const loadGuardrails = useCallback(() => {
    if (!accessToken) {
      return Promise.resolve();
    }
    return getGuardrailsList(accessToken)
      .then((response) => setGuardrails(compressionGuardrailsOf(response as GuardrailListResponse)))
      .catch((error) => {
        console.error("Failed to load compression guardrails:", error);
        NotificationsManager.fromBackend("Failed to load compression guardrails");
      })
      .finally(() => setIsLoading(false));
  }, [accessToken]);

  useEffect(() => {
    loadGuardrails();
  }, [loadGuardrails]);

  const handleModeChange = async (guardrail: GuardrailListItem, mode: CompressionMode) => {
    if (!accessToken) {
      return;
    }
    const nextDefaultOn = mode === "always";
    setPendingModeId(guardrail.guardrail_id);
    try {
      await updateGuardrailCall(accessToken, guardrail.guardrail_id, {
        litellm_params: { default_on: nextDefaultOn },
      });
      NotificationsManager.success(
        nextDefaultOn ? "Compression now runs on every request" : "Compression is now opt-in per request",
      );
      await loadGuardrails();
    } catch (error) {
      console.error("Failed to update compression guardrail:", error);
      NotificationsManager.fromBackend("Failed to update compression guardrail");
    } finally {
      setPendingModeId(null);
    }
  };

  const handleEditSettings = (guardrail: GuardrailListItem) => {
    router.push(guardrailDetailHref(guardrail.guardrail_id, "settings"));
  };

  const handleAdd = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!accessToken) {
      return;
    }
    if (!name.trim() || !apiBase.trim()) {
      setShowFieldErrors(true);
      return;
    }
    setIsSaving(true);
    try {
      await createGuardrailCall(accessToken, buildCompressionGuardrailPayload({ name, apiBase, defaultOn }));
      NotificationsManager.success("Compression guardrail created");
      setName("");
      setApiBase("");
      setDefaultOn(true);
      setShowFieldErrors(false);
      setIsAddFormOpen(false);
      await loadGuardrails();
    } catch (error) {
      console.error("Failed to create compression guardrail:", error);
      NotificationsManager.fromBackend("Failed to create compression guardrail");
    } finally {
      setIsSaving(false);
    }
  };

  const hasGuardrails = guardrails.length > 0;
  const isFormVisible = isAdmin && !isLoading && (!hasGuardrails || isAddFormOpen);

  return (
    <Card>
      <CardHeader>
        <CardTitle>Headroom prompt compression</CardTitle>
        <CardDescription>
          Headroom is a native LiteLLM guardrail that compresses your prompts before they reach the model, so you pay
          for fewer input tokens. The tokens it removes are priced and shown on the Usage tab as compression savings.{" "}
          <a
            href="https://docs.litellm.ai/docs/proxy/headroom"
            target="_blank"
            rel="noopener noreferrer"
            className="font-medium text-primary underline underline-offset-4"
          >
            Headroom setup docs
          </a>
        </CardDescription>
      </CardHeader>

      <CardContent className="space-y-4">
        {isLoading && <Skeleton className="h-14 w-full" />}

        {!isLoading && hasGuardrails && (
          <ul className="divide-y divide-border rounded-lg border border-border">
            {guardrails.map((guardrail) => (
              <CompressionEndpointRow
                key={guardrail.guardrail_id}
                guardrail={guardrail}
                canEdit={isAdmin}
                isPending={pendingModeId === guardrail.guardrail_id}
                onModeChange={handleModeChange}
                onEditSettings={handleEditSettings}
              />
            ))}
          </ul>
        )}

        {!isLoading && !hasGuardrails && !isFormVisible && (
          <p className="text-sm text-muted-foreground">
            No prompt compression endpoint is configured. An admin can add one to start saving on input tokens
          </p>
        )}

        {!isLoading && hasGuardrails && isAdmin && !isAddFormOpen && (
          <Button variant="ghost" size="sm" onClick={() => setIsAddFormOpen(true)}>
            <Plus />
            Add another endpoint
          </Button>
        )}

        {isFormVisible && (
          <form onSubmit={handleAdd} className="space-y-4 rounded-lg border border-border p-4">
            <div className="space-y-2">
              <Label htmlFor="compression-name">Name</Label>
              <Input
                id="compression-name"
                value={name}
                onChange={(event) => setName(event.target.value)}
                placeholder="headroom-compression"
                aria-invalid={showFieldErrors && !name.trim()}
              />
              {showFieldErrors && !name.trim() && <p className="text-xs text-destructive">Name is required</p>}
            </div>

            <div className="space-y-2">
              <Label htmlFor="compression-api-base">Headroom API base</Label>
              <Input
                id="compression-api-base"
                value={apiBase}
                onChange={(event) => setApiBase(event.target.value)}
                placeholder="https://your-headroom-endpoint"
                aria-invalid={showFieldErrors && !apiBase.trim()}
              />
              <p className="text-xs text-muted-foreground">
                Where your Headroom compression service is hosted; LiteLLM calls its /v1/compress endpoint
              </p>
              {showFieldErrors && !apiBase.trim() && <p className="text-xs text-destructive">API base is required</p>}
            </div>

            <div className="space-y-2">
              <Label htmlFor="compression-default-on">
                <Switch
                  id="compression-default-on"
                  checked={defaultOn}
                  onCheckedChange={(checked) => setDefaultOn(checked)}
                />
                Apply to all requests
              </Label>
              <p className="text-xs text-muted-foreground">
                Off means callers opt in per request. Applying compression to all requests is available to all users;
                enabling it selectively per key or team is a LiteLLM Enterprise feature.{" "}
                <a
                  href="https://www.litellm.ai/#pricing"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="font-medium text-primary underline underline-offset-4"
                >
                  Get a trial key
                </a>
              </p>
            </div>

            <div className="flex justify-end gap-2">
              {hasGuardrails && (
                <Button type="button" variant="ghost" onClick={() => setIsAddFormOpen(false)}>
                  Cancel
                </Button>
              )}
              <Button type="submit" disabled={isSaving}>
                {isSaving ? "Adding..." : "Add guardrail"}
              </Button>
            </div>
          </form>
        )}
      </CardContent>
    </Card>
  );
};

export default PromptCompressionTab;
