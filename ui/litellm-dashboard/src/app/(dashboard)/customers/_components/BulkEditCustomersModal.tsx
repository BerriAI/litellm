"use client";

import { useRef, useState, type FormEvent, type ReactNode } from "react";
import { useBudgetOptions } from "@/app/(dashboard)/hooks/budgets/useBudgetOptions";
import type { BulkUpdateCustomersResponse } from "@/app/(dashboard)/hooks/customers/customerApi";
import { useBulkUpdateCustomers } from "@/app/(dashboard)/hooks/customers/useCustomerMutations";
import type { EndUser } from "@/app/(dashboard)/hooks/customers/useCustomers";
import { useMCPAccessGroups } from "@/app/(dashboard)/hooks/mcpServers/useMCPAccessGroups";
import { useMCPServers } from "@/app/(dashboard)/hooks/mcpServers/useMCPServers";
import { useMCPToolsets } from "@/app/(dashboard)/hooks/mcpServers/useMCPToolsets";
import { useUserModels } from "@/app/(dashboard)/hooks/models/useModels";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { MultiSelect, type MultiSelectOption } from "@/components/shared/MultiSelect";
import { SearchSelect } from "@/components/shared/SearchSelect";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
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
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { extractProxyErrorMessage, unwrapProxyErrorMessage } from "@/lib/http/client";
import { toast } from "@/lib/toast";
import { buildBulkCustomerPayload, emptyBulkCustomerEdits, type BulkCustomerEdits } from "./bulkCustomerPayload";
import { hasSharedCustomerBudget } from "./customerPayload";

type BulkEditCustomersModalProps = {
  allCustomers: readonly EndUser[];
  selectedCustomers: EndUser[];
  open: boolean;
  onCancel: () => void;
  onCompleted: (successfulIds: string[]) => void;
};

type CatalogState = { isLoading: boolean; isError: boolean; refetch: () => unknown };

function CatalogFeedback({ label, catalog }: { label: string; catalog: CatalogState }) {
  if (catalog.isLoading)
    return (
      <p role="status" className="text-sm text-muted-foreground">
        Loading {label}…
      </p>
    );
  if (!catalog.isError) return null;
  return (
    <div className="flex items-center gap-2 text-sm text-destructive" role="alert">
      <span>{label} could not be loaded</span>
      <Button type="button" variant="outline" size="sm" onClick={() => void catalog.refetch()}>
        Retry {label}
      </Button>
    </div>
  );
}

function EditField({
  id,
  label,
  enabled,
  disabled,
  onEnabledChange,
  children,
}: {
  id: string;
  label: string;
  enabled: boolean;
  disabled: boolean;
  onEnabledChange: (enabled: boolean) => void;
  children?: ReactNode;
}) {
  return (
    <section className="space-y-3 rounded-lg border p-4">
      <div className="flex items-center gap-2">
        <Checkbox id={`${id}-enabled`} checked={enabled} disabled={disabled} onCheckedChange={onEnabledChange} />
        <Label htmlFor={`${id}-enabled`}>Update {label}</Label>
      </div>
      {children}
    </section>
  );
}

function ArrayEditField({
  id,
  label,
  edit,
  options,
  catalog,
  disabled,
  description,
  onChange,
}: {
  id: string;
  label: string;
  edit: { enabled: boolean; value: string[] };
  options: MultiSelectOption[];
  catalog: CatalogState;
  disabled: boolean;
  description: string;
  onChange: (edit: { enabled: boolean; value: string[] }) => void;
}) {
  const unavailable = catalog.isLoading || catalog.isError;
  return (
    <EditField
      id={id}
      label={label}
      enabled={edit.enabled}
      disabled={disabled || (unavailable && !edit.enabled)}
      onEnabledChange={(enabled) => onChange({ ...edit, enabled })}
    >
      <CatalogFeedback label={label} catalog={catalog} />
      {edit.enabled && (
        <div className="space-y-2">
          <Label htmlFor={id}>{label}</Label>
          <MultiSelect
            id={id}
            options={options}
            value={edit.value}
            placeholder={label}
            disabled={disabled || unavailable}
            onValueChange={(value) => onChange({ ...edit, value })}
          />
          <p className="text-sm text-muted-foreground">{description}</p>
        </div>
      )}
    </EditField>
  );
}

const regionOptions = [
  { value: "eu", label: "EU" },
  { value: "us", label: "US" },
];
const budgetModes = [
  { value: "link", label: "Link an existing budget" },
  { value: "max", label: "Set a maximum budget" },
];

function BudgetEditField({
  edit,
  catalog,
  hasSharedBudget,
  disabled,
  onChange,
}: {
  edit: BulkCustomerEdits["budget"];
  catalog: ReturnType<typeof useBudgetOptions>;
  hasSharedBudget: boolean;
  disabled: boolean;
  onChange: (edit: BulkCustomerEdits["budget"]) => void;
}) {
  const budgetUnavailable = edit.mode === "link" && (catalog.isLoading || catalog.isError);
  return (
    <EditField
      id="bulk-budget"
      label="budget"
      enabled={edit.enabled}
      disabled={disabled || (budgetUnavailable && !edit.enabled)}
      onEnabledChange={(enabled) => onChange({ ...edit, enabled })}
    >
      <Label htmlFor="bulk-budget-mode">Budget update</Label>
      <Select
        items={budgetModes}
        value={edit.mode}
        disabled={disabled}
        onValueChange={(mode) => {
          if (mode === "link" || mode === "max") onChange({ ...edit, mode });
        }}
      >
        <SelectTrigger id="bulk-budget-mode" className="w-full">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {budgetModes.map((mode) => (
            <SelectItem key={mode.value} value={mode.value} disabled={mode.value === "max" && hasSharedBudget}>
              {mode.label}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      {hasSharedBudget && (
        <p role="note" className="text-sm text-muted-foreground">
          Some selected customers share a budget with other customers. Changing a shared budget maximum can affect
          unselected customers, so choose an existing budget instead.
        </p>
      )}
      {edit.mode === "link" && <CatalogFeedback label="Budgets" catalog={catalog} />}
      {edit.enabled && edit.mode === "link" && (
        <div className="space-y-2">
          <Label htmlFor="bulk-budget-id">Budget</Label>
          <SearchSelect
            inputId="bulk-budget-id"
            aria-label="Budget"
            options={(catalog.data ?? []).map((budget) => ({ label: budget.budget_id, value: budget.budget_id }))}
            value={edit.budgetId}
            disabled={disabled || budgetUnavailable}
            allowClear={false}
            onValueChange={(budgetId) => onChange({ ...edit, budgetId: budgetId ?? "" })}
          />
          <p className="text-sm text-muted-foreground">Replace the budget link for every selected customer.</p>
        </div>
      )}
      {edit.enabled && edit.mode === "max" && (
        <div className="space-y-2">
          <Label htmlFor="bulk-max-budget">Max Budget (USD)</Label>
          <Input
            id="bulk-max-budget"
            type="number"
            min={0}
            step="any"
            value={edit.maxBudget}
            disabled={disabled || hasSharedBudget}
            onChange={(event) => onChange({ ...edit, maxBudget: event.target.value })}
          />
          <p className="text-sm text-muted-foreground">Enter a maximum greater than zero.</p>
        </div>
      )}
    </EditField>
  );
}

function BulkEditSession({
  selectedCustomers,
  allCustomers,
  onCancel,
  onCompleted,
}: Omit<BulkEditCustomersModalProps, "open">) {
  const [customers, setCustomers] = useState(() => selectedCustomers);
  const [edits, setEdits] = useState(emptyBulkCustomerEdits);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<BulkUpdateCustomersResponse | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const inFlight = useRef(false);
  const { accessToken } = useAuthorized();
  const budgets = useBudgetOptions(accessToken);
  const models = useUserModels();
  const servers = useMCPServers();
  const groups = useMCPAccessGroups();
  const toolsets = useMCPToolsets();
  const mutation = useBulkUpdateCustomers();
  const pending = submitting || mutation.isPending;
  const hasSharedBudget = customers.some((customer) => hasSharedCustomerBudget(customer, allCustomers));
  const modelOptions = (models.data ?? []).map((model) => ({ label: model, value: model }));
  const modelsUnavailable = models.isLoading || models.isError;
  const actionLabel = results ? "Retry failed customers" : "Apply changes";

  const update = <Field extends keyof BulkCustomerEdits>(field: Field, value: BulkCustomerEdits[Field]) => {
    setEdits((current) => ({ ...current, [field]: value }));
    setError(null);
  };
  const cancel = () => {
    if (!inFlight.current) onCancel();
  };
  const submit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (inFlight.current) return;
    const payload = buildBulkCustomerPayload(customers, edits, servers.data ?? [], allCustomers);
    if (!payload.ok) {
      setError(payload.error);
      return;
    }
    const unavailable = [
      { enabled: edits.budget.enabled && edits.budget.mode === "link", catalog: budgets, label: "Budgets" },
      { enabled: edits.models.enabled || edits.defaultModel.enabled, catalog: models, label: "Models" },
      { enabled: edits.mcpServers.enabled, catalog: servers, label: "MCP servers" },
      { enabled: edits.mcpAccessGroups.enabled, catalog: groups, label: "MCP access groups" },
      { enabled: edits.mcpToolsets.enabled, catalog: toolsets, label: "MCP toolsets" },
    ].find(({ enabled, catalog }) => enabled && (catalog.isLoading || catalog.isError));
    if (unavailable) {
      setError(`${unavailable.label} must finish loading before applying changes`);
      return;
    }
    inFlight.current = true;
    setSubmitting(true);
    setError(null);
    try {
      const response = await mutation.mutateAsync(payload.payload);
      const successfulIds = response.results.filter((result) => result.success).map((result) => result.user_id);
      const remaining = customers.filter((customer) => !successfulIds.includes(customer.user_id));
      setCustomers(remaining);
      setResults(response);
      if (successfulIds.length > 0) onCompleted(successfulIds);
      if (remaining.length === 0) {
        toast.success(`${response.successful_updates} customer${response.successful_updates === 1 ? "" : "s"} updated`);
        onCancel();
      }
    } catch (failure) {
      setError(extractProxyErrorMessage(failure));
    } finally {
      inFlight.current = false;
      setSubmitting(false);
    }
  };

  return (
    <Dialog
      open
      onOpenChange={(nextOpen) => {
        if (!nextOpen) cancel();
      }}
    >
      <DialogContent className="max-h-[calc(100dvh-2rem)] overflow-y-auto sm:max-w-2xl" showCloseButton={!pending}>
        <DialogHeader>
          <DialogTitle>Edit selected customers</DialogTitle>
          <DialogDescription>
            {customers.length} customers to update. Check each field you want to replace; unchecked fields stay
            unchanged.
          </DialogDescription>
        </DialogHeader>
        <section aria-label="Customers to update" className="rounded-lg border p-3">
          <ul className="max-h-24 overflow-y-auto text-sm">
            {customers.map((customer) => (
              <li key={customer.user_id}>
                <code>{customer.user_id}</code>
              </li>
            ))}
          </ul>
        </section>
        <form aria-label="Bulk customer edits" onSubmit={submit} className="space-y-4">
          <BudgetEditField
            edit={edits.budget}
            catalog={budgets}
            hasSharedBudget={hasSharedBudget}
            disabled={pending}
            onChange={(edit) => update("budget", edit)}
          />
          <ArrayEditField
            id="bulk-models"
            label="Allowed models"
            edit={edits.models}
            options={modelOptions}
            catalog={models}
            disabled={pending}
            description="An empty selection allows all models the key can use."
            onChange={(edit) => update("models", edit)}
          />
          <EditField
            id="bulk-region"
            label="allowed region"
            enabled={edits.region.enabled}
            disabled={pending}
            onEnabledChange={(enabled) => update("region", { ...edits.region, enabled })}
          >
            {edits.region.enabled && (
              <div className="space-y-2">
                <Label htmlFor="bulk-region">Allowed region</Label>
                <Select
                  items={regionOptions}
                  value={edits.region.value}
                  disabled={pending}
                  onValueChange={(value) => {
                    if (value === "eu" || value === "us") update("region", { ...edits.region, value });
                  }}
                >
                  <SelectTrigger id="bulk-region" className="w-full">
                    <SelectValue placeholder="Choose a region" />
                  </SelectTrigger>
                  <SelectContent>
                    {regionOptions.map((region) => (
                      <SelectItem key={region.value} value={region.value}>
                        {region.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            )}
          </EditField>
          <EditField
            id="bulk-default-model"
            label="default model"
            enabled={edits.defaultModel.enabled}
            disabled={pending || (modelsUnavailable && !edits.defaultModel.enabled)}
            onEnabledChange={(enabled) => update("defaultModel", { ...edits.defaultModel, enabled })}
          >
            <CatalogFeedback label="Default models" catalog={models} />
            {edits.defaultModel.enabled && (
              <div className="space-y-2">
                <Label htmlFor="bulk-default-model">Default model</Label>
                <SearchSelect
                  inputId="bulk-default-model"
                  aria-label="Default model"
                  options={modelOptions}
                  value={edits.defaultModel.value}
                  disabled={pending || models.isLoading || models.isError}
                  allowClear={false}
                  onValueChange={(value) => update("defaultModel", { ...edits.defaultModel, value: value ?? "" })}
                />
              </div>
            )}
          </EditField>
          <ArrayEditField
            id="bulk-mcp-servers"
            label="MCP servers"
            edit={edits.mcpServers}
            catalog={servers}
            disabled={pending}
            options={(servers.data ?? []).map((server) => ({
              value: server.server_id,
              label: server.alias || server.server_name || server.server_id,
            }))}
            description="An empty selection clears direct server grants. Tool grants for servers outside this selection are removed. Clearing all customer MCP grants removes the customer restriction."
            onChange={(edit) => update("mcpServers", edit)}
          />
          <ArrayEditField
            id="bulk-mcp-groups"
            label="MCP access groups"
            edit={edits.mcpAccessGroups}
            catalog={groups}
            disabled={pending}
            options={(groups.data ?? []).map((group) => ({ value: group, label: group }))}
            description="An empty selection clears access group grants. Other MCP permissions stay unchanged. Clearing all customer MCP grants removes the customer restriction."
            onChange={(edit) => update("mcpAccessGroups", edit)}
          />
          <ArrayEditField
            id="bulk-mcp-toolsets"
            label="MCP toolsets"
            edit={edits.mcpToolsets}
            catalog={toolsets}
            disabled={pending}
            options={(toolsets.data ?? []).map((toolset) => ({
              value: toolset.toolset_id,
              label: toolset.toolset_name,
            }))}
            description="An empty selection removes the customer toolset restriction. Other MCP permissions stay unchanged."
            onChange={(edit) => update("mcpToolsets", edit)}
          />
          {results && (
            <Alert aria-label="Bulk update results">
              <AlertTitle>
                {results.successful_updates} succeeded, {results.failed_updates} failed out of {results.total_requested}
              </AlertTitle>
              <AlertDescription>
                <p>Only failed customers remain for retry.</p>
                <ul>
                  {results.results
                    .filter((result) => !result.success)
                    .map((result) => (
                      <li key={result.user_id}>
                        <strong>{result.user_id}</strong>: {unwrapProxyErrorMessage(result.error ?? "Update failed")}
                      </li>
                    ))}
                </ul>
              </AlertDescription>
            </Alert>
          )}
          {error && (
            <Alert variant="destructive" aria-label="Bulk update error">
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}
          <DialogFooter>
            <Button type="button" variant="outline" disabled={pending} onClick={cancel}>
              Cancel
            </Button>
            <Button type="submit" disabled={pending}>
              {pending ? "Applying changes…" : actionLabel}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}

export function BulkEditCustomersModal({ open, ...props }: BulkEditCustomersModalProps) {
  if (!open) return null;
  return <BulkEditSession {...props} />;
}
