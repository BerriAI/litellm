"use client";

import React, { useCallback, useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { Plus, Settings2, Trash2 } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { Switch } from "@/components/ui/switch";
import {
  createGuardrailCall,
  deleteGuardrailCall,
  getGuardrailsList,
  updateGuardrailCall,
} from "@/components/networking";
import NotificationsManager from "@/components/molecules/notifications_manager";
import DeleteResourceModal from "@/components/common_components/DeleteResourceModal";
import { guardrailDetailHref } from "@/app/(dashboard)/guardrails/detailNavigation";
import { isAdminRole } from "@/utils/roles";
import {
  buildCompressionGuardrailPayload,
  CompressionGuardrailInput,
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
  onDelete: (guardrail: GuardrailListItem) => void;
}

const CompressionEndpointRow: React.FC<CompressionEndpointRowProps> = ({
  guardrail,
  canEdit,
  isPending,
  onModeChange,
  onEditSettings,
  onDelete,
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

        {isEditable && (
          <Button
            variant="ghost"
            size="sm"
            aria-label={`Delete ${name}`}
            className="text-muted-foreground hover:text-destructive"
            onClick={() => onDelete(guardrail)}
          >
            <Trash2 />
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

interface AddEndpointFormProps {
  isSaving: boolean;
  canCancel: boolean;
  onCancel: () => void;
  onSubmit: (input: CompressionGuardrailInput) => void;
}

const AddEndpointForm: React.FC<AddEndpointFormProps> = ({ isSaving, canCancel, onCancel, onSubmit }) => {
  const [name, setName] = useState<string>("");
  const [apiBase, setApiBase] = useState<string>("");
  const [defaultOn, setDefaultOn] = useState<boolean>(true);
  const [showFieldErrors, setShowFieldErrors] = useState<boolean>(false);

  const isNameMissing = !name.trim();
  const isApiBaseMissing = !apiBase.trim();

  const handleSubmit = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (isNameMissing || isApiBaseMissing) {
      setShowFieldErrors(true);
      return;
    }
    onSubmit({ name, apiBase, defaultOn });
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4 rounded-lg border border-border p-4">
      <div className="space-y-2">
        <Label htmlFor="compression-name">Name</Label>
        <Input
          id="compression-name"
          value={name}
          onChange={(event) => setName(event.target.value)}
          placeholder="headroom-compression"
          aria-invalid={showFieldErrors && isNameMissing}
        />
        {showFieldErrors && isNameMissing && <p className="text-xs text-destructive">Name is required</p>}
      </div>

      <div className="space-y-2">
        <Label htmlFor="compression-api-base">Headroom API base</Label>
        <Input
          id="compression-api-base"
          value={apiBase}
          onChange={(event) => setApiBase(event.target.value)}
          placeholder="https://your-headroom-endpoint"
          aria-invalid={showFieldErrors && isApiBaseMissing}
        />
        <p className="text-xs text-muted-foreground">
          Where your Headroom compression service is hosted; LiteLLM calls its /v1/compress endpoint
        </p>
        {showFieldErrors && isApiBaseMissing && <p className="text-xs text-destructive">API base is required</p>}
      </div>

      <div className="space-y-2">
        <Label htmlFor="compression-default-on">
          <Switch id="compression-default-on" checked={defaultOn} onCheckedChange={setDefaultOn} />
          Apply to all requests
        </Label>
        <p className="text-xs text-muted-foreground">
          Off means callers opt in per request. Applying compression to all requests is available to all users; enabling
          it selectively per key or team is a LiteLLM Enterprise feature.{" "}
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
        {canCancel && (
          <Button type="button" variant="ghost" onClick={onCancel}>
            Cancel
          </Button>
        )}
        <Button type="submit" disabled={isSaving}>
          {isSaving ? "Adding..." : "Add guardrail"}
        </Button>
      </div>
    </form>
  );
};

interface DeleteEndpointModalProps {
  guardrail: GuardrailListItem | null;
  isDeleting: boolean;
  onCancel: () => void;
  onConfirm: () => void;
}

const DeleteEndpointModal: React.FC<DeleteEndpointModalProps> = ({ guardrail, isDeleting, onCancel, onConfirm }) => {
  const isAlwaysOn = guardrail !== null && modeOf(guardrail) === "always";

  return (
    <DeleteResourceModal
      isOpen={guardrail !== null}
      title="Delete compression guardrail"
      alertMessage={
        isAlwaysOn
          ? "Every request is compressed by this guardrail today. Deleting it stops compression immediately and input token costs go back up"
          : "Requests that ask for this guardrail by name will fail once it is deleted"
      }
      message={`Are you sure you want to delete guardrail: ${guardrail?.guardrail_name ?? ""}? This action cannot be undone.`}
      resourceInformationTitle="Guardrail Information"
      resourceInformation={[
        { label: "Name", value: guardrail?.guardrail_name },
        { label: "ID", value: guardrail?.guardrail_id, code: true },
        { label: "API base", value: guardrail?.litellm_params?.api_base },
        { label: "Applies to", value: isAlwaysOn ? MODE_LABELS.always : MODE_LABELS.opt_in },
      ]}
      onCancel={onCancel}
      onOk={onConfirm}
      confirmLoading={isDeleting}
    />
  );
};

const PromptCompressionTab: React.FC<PromptCompressionTabProps> = ({ accessToken, userRole }) => {
  const router = useRouter();
  const isAdmin = userRole ? isAdminRole(userRole) : false;

  const [guardrails, setGuardrails] = useState<GuardrailListItem[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [isSaving, setIsSaving] = useState<boolean>(false);
  const [pendingModeId, setPendingModeId] = useState<string | null>(null);
  const [guardrailToDelete, setGuardrailToDelete] = useState<GuardrailListItem | null>(null);
  const [isDeleting, setIsDeleting] = useState<boolean>(false);
  const [isAddFormOpen, setIsAddFormOpen] = useState<boolean>(false);

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

  const handleDeleteConfirm = async () => {
    if (!accessToken || !guardrailToDelete) {
      return;
    }
    setIsDeleting(true);
    try {
      await deleteGuardrailCall(accessToken, guardrailToDelete.guardrail_id);
      NotificationsManager.success("Compression guardrail deleted");
      setGuardrailToDelete(null);
      await loadGuardrails();
    } catch (error) {
      console.error("Failed to delete compression guardrail:", error);
      NotificationsManager.fromBackend("Failed to delete compression guardrail");
    } finally {
      setIsDeleting(false);
    }
  };

  const handleAdd = async (input: CompressionGuardrailInput) => {
    if (!accessToken) {
      return;
    }
    setIsSaving(true);
    try {
      await createGuardrailCall(accessToken, buildCompressionGuardrailPayload(input));
      NotificationsManager.success("Compression guardrail created");
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
  const isSettled = !isLoading;
  const wantsAddForm = !hasGuardrails || isAddFormOpen;
  const isFormVisible = isAdmin && isSettled && wantsAddForm;
  const isListVisible = isSettled && hasGuardrails;
  const isEmptyStateVisible = isSettled && !hasGuardrails && !isFormVisible;
  const isAddAnotherVisible = isListVisible && isAdmin && !isAddFormOpen;

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
        {!isSettled && <Skeleton className="h-14 w-full" />}

        {isListVisible && (
          <ul className="divide-y divide-border rounded-lg border border-border">
            {guardrails.map((guardrail) => (
              <CompressionEndpointRow
                key={guardrail.guardrail_id}
                guardrail={guardrail}
                canEdit={isAdmin}
                isPending={pendingModeId === guardrail.guardrail_id}
                onModeChange={handleModeChange}
                onEditSettings={handleEditSettings}
                onDelete={setGuardrailToDelete}
              />
            ))}
          </ul>
        )}

        {isEmptyStateVisible && (
          <p className="text-sm text-muted-foreground">
            No prompt compression endpoint is configured. An admin can add one to start saving on input tokens
          </p>
        )}

        {isAddAnotherVisible && (
          <Button variant="ghost" size="sm" onClick={() => setIsAddFormOpen(true)}>
            <Plus />
            Add another endpoint
          </Button>
        )}

        {isFormVisible && (
          <AddEndpointForm
            isSaving={isSaving}
            canCancel={hasGuardrails}
            onCancel={() => setIsAddFormOpen(false)}
            onSubmit={handleAdd}
          />
        )}

        <DeleteEndpointModal
          guardrail={guardrailToDelete}
          isDeleting={isDeleting}
          onCancel={() => setGuardrailToDelete(null)}
          onConfirm={handleDeleteConfirm}
        />
      </CardContent>
    </Card>
  );
};

export default PromptCompressionTab;
