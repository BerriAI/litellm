"use client";

import { useState, type FormEvent } from "react";
import { useTranslation } from "react-i18next";
import { toast } from "sonner";
import { Trash2 } from "lucide-react";
import { fetchClient } from "@/lib/http/api";
import { extractProxyErrorMessage } from "@/lib/http/client";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  buildCustomerHierarchyPatch,
  customerEntityId,
  customerEntityLabel,
  customerHierarchyFields,
  type CustomerHierarchyEntity,
  type CustomerHierarchyFields,
  type CustomerHierarchyPatch,
  type Organization,
} from "./customerHierarchyForms";

interface Props {
  entity: CustomerHierarchyEntity;
  organization: Organization;
  onSaved: () => void;
  onDeleted: () => void;
}

const saveEntity = async (id: string, patch: CustomerHierarchyPatch): Promise<void> => {
  switch (patch.kind) {
    case "organization":
      await fetchClient.PATCH("/management/v1/customer_organizations/{customer_org_id}", {
        params: { path: { customer_org_id: id } },
        body: patch.body,
      });
      return;
    case "team":
      await fetchClient.PATCH("/management/v1/customer_teams/{customer_team_id}", {
        params: { path: { customer_team_id: id } },
        body: patch.body,
      });
      return;
    case "member":
      await fetchClient.PATCH("/management/v1/customer_members/{customer_member_id}", {
        params: { path: { customer_member_id: id } },
        body: patch.body,
      });
  }
};

const deleteEntity = async (entity: CustomerHierarchyEntity): Promise<void> => {
  const id = customerEntityId(entity);
  switch (entity.kind) {
    case "organization":
      await fetchClient.DELETE("/management/v1/customer_organizations/{customer_org_id}", {
        params: { path: { customer_org_id: id } },
      });
      return;
    case "team":
      await fetchClient.DELETE("/management/v1/customer_teams/{customer_team_id}", {
        params: { path: { customer_team_id: id } },
      });
      return;
    case "member":
      await fetchClient.DELETE("/management/v1/customer_members/{customer_member_id}", {
        params: { path: { customer_member_id: id } },
      });
  }
};

const KIND_LABELS = {
  organization: "customerPortal.hierarchy.organization",
  team: "customerPortal.hierarchy.team",
  member: "customerPortal.hierarchy.member",
} as const;

const ERROR_LABELS = {
  invalidNumber: "customerPortal.hierarchy.invalidNumber",
  invalidExpiry: "customerPortal.hierarchy.invalidExpiry",
  budgetRequired: "customerPortal.hierarchy.budgetRequired",
} as const;

function SettingsForm({ entity, organization, onSaved, onDeleted }: Props) {
  const { t } = useTranslation();
  const initial = customerHierarchyFields(entity);
  const [fields, setFields] = useState(initial);
  const [pending, setPending] = useState<"save" | "delete" | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [deleteOpen, setDeleteOpen] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);
  const dirty = JSON.stringify(fields) !== JSON.stringify(initial);
  const id = customerEntityId(entity);
  const kind = t(KIND_LABELS[entity.kind]);
  const update = <K extends keyof CustomerHierarchyFields>(key: K, value: CustomerHierarchyFields[K]) =>
    setFields((current) => ({ ...current, [key]: value }));
  const templateLabel = (templateId: string): string => {
    if (!templateId) return t("customerPortal.hierarchy.noTemplate");
    const template = organization.default_budgets.find((budget) => budget.budget_id === templateId);
    if (!template) return t("customerPortal.hierarchy.unknownTemplate", { id: templateId });
    if (templateId === organization.default_budget_id)
      return t("customerPortal.hierarchy.defaultTemplate", { name: template.name });
    return template.name;
  };

  const save = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const result = buildCustomerHierarchyPatch(entity, fields);
    if (!result.ok) {
      setError(t(ERROR_LABELS[result.error]));
      return;
    }
    setPending("save");
    setError(null);
    try {
      await saveEntity(id, result.patch);
      toast.success(t("customerPortal.hierarchy.saved"));
      onSaved();
    } catch (cause) {
      setError(extractProxyErrorMessage(cause));
    } finally {
      setPending(null);
    }
  };

  const remove = async () => {
    setPending("delete");
    setDeleteError(null);
    try {
      await deleteEntity(entity);
      setDeleteOpen(false);
      onDeleted();
    } catch (cause) {
      setDeleteError(extractProxyErrorMessage(cause));
    } finally {
      setPending(null);
    }
  };

  const numericFields = [
    ["maxBudget", "customerPortal.hierarchy.maxBudget", "any"],
    ["softBudget", "customerPortal.hierarchy.softBudget", "any"],
    ["tpm", "customerPortal.hierarchy.tpm", "1"],
    ["rpm", "customerPortal.hierarchy.rpm", "1"],
  ] as const;

  return (
    <div className="max-w-3xl space-y-8">
      <form onSubmit={(event) => void save(event)} className="space-y-7">
        <fieldset disabled={pending !== null} className="space-y-7">
          <section className="space-y-4" aria-labelledby="hierarchy-general-title">
            <div>
              <h2 id="hierarchy-general-title" className="text-base font-medium">
                {t("customerPortal.hierarchy.profileTitle")}
              </h2>
              <p className="mt-1 text-sm text-muted-foreground">{t("customerPortal.hierarchy.profileHint")}</p>
            </div>
            <div className="grid gap-4 sm:grid-cols-2">
              <div className="space-y-1.5">
                <Label htmlFor="hierarchy-alias">{t("customerPortal.hierarchy.alias")}</Label>
                <Input
                  id="hierarchy-alias"
                  value={fields.alias}
                  onChange={(event) => update("alias", event.target.value)}
                />
              </div>
              <div className="space-y-1.5">
                <Label htmlFor="hierarchy-id">{t("customerPortal.hierarchy.identifier")}</Label>
                <Input id="hierarchy-id" value={id} readOnly className="bg-muted font-mono text-xs" />
              </div>
            </div>
          </section>

          <section className="space-y-4 border-t pt-6" aria-labelledby="hierarchy-budget-title">
            <div>
              <h2 id="hierarchy-budget-title" className="text-base font-medium">
                {t("customerPortal.hierarchy.budgetTitle")}
              </h2>
              <p className="mt-1 text-sm text-muted-foreground">{t("customerPortal.hierarchy.budgetHint")}</p>
            </div>
            {entity.kind === "member" && (
              <div className="max-w-sm space-y-1.5">
                <Label htmlFor="hierarchy-template">{t("customerPortal.hierarchy.template")}</Label>
                <Select
                  value={fields.templateId || "__none__"}
                  onValueChange={(value) => update("templateId", value === "__none__" ? "" : value ?? "")}
                >
                  <SelectTrigger id="hierarchy-template" className="w-full">
                    <SelectValue>{templateLabel(fields.templateId)}</SelectValue>
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="__none__">{t("customerPortal.hierarchy.noTemplate")}</SelectItem>
                    {organization.default_budgets.map((template) => (
                      <SelectItem key={template.budget_id} value={template.budget_id}>
                        {templateLabel(template.budget_id)}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            )}
            <div className="space-y-1.5">
              <div className="flex items-center gap-2">
                <Checkbox
                  id="hierarchy-budget-enabled"
                  checked={fields.budgetEnabled}
                  onCheckedChange={(checked) => update("budgetEnabled", checked)}
                />
                <Label htmlFor="hierarchy-budget-enabled">
                  {t(
                    entity.kind === "member"
                      ? "customerPortal.hierarchy.overrideEnabled"
                      : "customerPortal.hierarchy.budgetEnabled",
                  )}
                </Label>
              </div>
              {entity.kind === "member" && (
                <p className="text-xs text-muted-foreground">{t("customerPortal.hierarchy.overrideHint")}</p>
              )}
            </div>
            {fields.budgetEnabled && (
              <div className="space-y-4">
                <div className="grid gap-4 sm:grid-cols-2">
                  {numericFields.map(([name, label, step]) => (
                    <div className="space-y-1.5" key={name}>
                      <Label htmlFor={`hierarchy-${name}`}>{t(label)}</Label>
                      <Input
                        id={`hierarchy-${name}`}
                        type="number"
                        min="0"
                        step={step}
                        value={fields[name]}
                        onChange={(event) => update(name, event.target.value)}
                      />
                    </div>
                  ))}
                  <div className="space-y-1.5">
                    <Label htmlFor="hierarchy-duration">{t("customerPortal.hierarchy.budgetDuration")}</Label>
                    <Input
                      id="hierarchy-duration"
                      value={fields.budgetDuration}
                      onChange={(event) => update("budgetDuration", event.target.value)}
                      aria-describedby="hierarchy-duration-hint"
                    />
                    <p id="hierarchy-duration-hint" className="text-xs text-muted-foreground">
                      {t("customerPortal.hierarchy.budgetDurationHint")}
                    </p>
                  </div>
                  {entity.kind === "member" && (
                    <div className="space-y-1.5">
                      <Label htmlFor="hierarchy-expiry">{t("customerPortal.hierarchy.overrideExpiry")}</Label>
                      <Input
                        id="hierarchy-expiry"
                        type="datetime-local"
                        value={fields.overrideExpiry}
                        onChange={(event) => update("overrideExpiry", event.target.value)}
                        aria-describedby="hierarchy-expiry-hint"
                      />
                      <p id="hierarchy-expiry-hint" className="text-xs text-muted-foreground">
                        {t("customerPortal.hierarchy.overrideExpiryHint")}
                      </p>
                    </div>
                  )}
                </div>
                <div className="space-y-1.5">
                  <Label htmlFor="hierarchy-models">{t("customerPortal.hierarchy.allowedModels")}</Label>
                  <Textarea
                    id="hierarchy-models"
                    value={fields.allowedModels}
                    onChange={(event) => update("allowedModels", event.target.value)}
                    aria-describedby="hierarchy-models-hint"
                    className="min-h-20 font-mono text-xs"
                  />
                  <p id="hierarchy-models-hint" className="text-xs text-muted-foreground">
                    {t("customerPortal.hierarchy.allowedModelsHint")}
                  </p>
                </div>
              </div>
            )}
          </section>
        </fieldset>
        {error && (
          <p role="alert" className="text-sm text-destructive">
            {error}
          </p>
        )}
        <div className="flex justify-end gap-2 border-t pt-4">
          <Button
            type="button"
            variant="outline"
            disabled={pending !== null || !dirty}
            onClick={() => {
              setFields(initial);
              setError(null);
            }}
          >
            {t("customerPortal.hierarchy.reset")}
          </Button>
          <Button type="submit" disabled={pending !== null || !dirty}>
            {t(pending === "save" ? "customerPortal.hierarchy.saving" : "customerPortal.hierarchy.save")}
          </Button>
        </div>
      </form>

      <section
        className="flex flex-wrap items-start justify-between gap-4 border-t pt-6"
        aria-labelledby="hierarchy-delete-title"
      >
        <div className="max-w-xl space-y-1">
          <h2 id="hierarchy-delete-title" className="text-sm font-medium">
            {t("customerPortal.hierarchy.deleteTitle", { kind })}
          </h2>
          <p className="text-xs text-muted-foreground">{t("customerPortal.hierarchy.deleteHint", { kind })}</p>
          {entity.kind !== "member" && (
            <p className="text-xs text-muted-foreground">{t("customerPortal.hierarchy.deleteParentHint")}</p>
          )}
        </div>
        <Button
          type="button"
          variant="outline"
          className="text-destructive"
          disabled={pending !== null}
          onClick={() => {
            setDeleteError(null);
            setDeleteOpen(true);
          }}
        >
          <Trash2 className="size-3.5" />
          {t("customerPortal.hierarchy.deleteTitle", { kind })}
        </Button>
      </section>
      <Dialog
        open={deleteOpen}
        onOpenChange={(open) => {
          if (pending !== "delete") setDeleteOpen(open);
        }}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>
              {t("customerPortal.hierarchy.deleteConfirmTitle", { name: customerEntityLabel(entity) })}
            </DialogTitle>
            <DialogDescription>{t("customerPortal.hierarchy.deleteConfirmHint", { kind })}</DialogDescription>
          </DialogHeader>
          <p className="font-mono text-xs text-muted-foreground">{id}</p>
          {entity.kind !== "member" && (
            <p className="text-sm text-muted-foreground">{t("customerPortal.hierarchy.deleteParentHint")}</p>
          )}
          {deleteError && (
            <p role="alert" className="text-sm text-destructive">
              {deleteError}
            </p>
          )}
          <DialogFooter>
            <Button
              type="button"
              variant="outline"
              disabled={pending === "delete"}
              onClick={() => setDeleteOpen(false)}
            >
              {t("customerPortal.hierarchy.cancel")}
            </Button>
            <Button type="button" variant="destructive" disabled={pending === "delete"} onClick={() => void remove()}>
              {t(
                pending === "delete" ? "customerPortal.hierarchy.deleting" : "customerPortal.hierarchy.deleteConfirm",
                { kind },
              )}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}

export default function CustomerHierarchySettings(props: Props) {
  return (
    <SettingsForm
      key={`${props.entity.kind}:${customerEntityId(props.entity)}:${props.entity.record.updated_at}`}
      {...props}
    />
  );
}
