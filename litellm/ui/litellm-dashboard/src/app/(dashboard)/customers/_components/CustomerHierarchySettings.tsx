"use client";

import { useState, type FormEvent } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { toast } from "sonner";
import { Trash2 } from "lucide-react";
import { fetchClient } from "@/lib/http/api";
import { extractProxyErrorMessage } from "@/lib/http/client";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import DeleteResourceModal from "@/components/common_components/DeleteResourceModal";
import {
  buildCustomerHierarchyPatch,
  customerEntityId,
  customerHierarchyFields,
  CUSTOMER_HIERARCHY_FORM_ERRORS,
  type CustomerHierarchyEntity,
  type CustomerHierarchyFields,
  type CustomerHierarchyPatch,
  type Organization,
} from "./customerHierarchyForms";

import CustomerHierarchyBudgetFields from "./CustomerHierarchyBudgetFields";

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
        params: { path: { customer_org_id: id }, query: { cascade: true } },
      });
      return;
    case "team":
      await fetchClient.DELETE("/management/v1/customer_teams/{customer_team_id}", {
        params: { path: { customer_team_id: id }, query: { cascade: true } },
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

const DELETE_MESSAGES = {
  organization: "customerPortal.hierarchy.deleteOrganizationMessage",
  team: "customerPortal.hierarchy.deleteTeamMessage",
  member: "customerPortal.hierarchy.deleteMemberMessage",
} as const;

function SettingsForm({ entity, organization, onSaved, onDeleted }: Props) {
  const queryClient = useQueryClient();
  const { t } = useTranslation();
  const initial = customerHierarchyFields(entity);
  const [fields, setFields] = useState(initial);
  const [pending, setPending] = useState<"save" | "delete" | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [deleteOpen, setDeleteOpen] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);
  const dirty = JSON.stringify(fields) !== JSON.stringify(initial);
  const id = customerEntityId(entity);
  const kindLabel = t(KIND_LABELS[entity.kind]);
  const kind = kindLabel.toLocaleLowerCase();
  const deleteMessage = t(DELETE_MESSAGES[entity.kind]);
  const deleteInformation = [
    { label: t("customerPortal.hierarchy.alias"), value: entity.record.alias },
    { label: t("customerPortal.hierarchy.identifier"), value: id, code: true },
  ];
  const update = <K extends keyof CustomerHierarchyFields>(key: K, value: CustomerHierarchyFields[K]) =>
    setFields((current) => ({ ...current, [key]: value }));
  const save = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const result = buildCustomerHierarchyPatch(entity, fields);
    if (!result.ok) {
      setError(t(CUSTOMER_HIERARCHY_FORM_ERRORS[result.error]));
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
      void queryClient.invalidateQueries({ queryKey: ["get", "/management/v1/customer_portal/limits"] });
      setDeleteOpen(false);
      onDeleted();
    } catch (cause) {
      setDeleteError(extractProxyErrorMessage(cause));
    } finally {
      setPending(null);
    }
  };

  return (
    <div className="w-full space-y-8">
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

          <CustomerHierarchyBudgetFields
            kind={entity.kind}
            organization={organization}
            fields={fields}
            onChange={update}
            idPrefix="customer-hierarchy"
            operation="edit"
          />
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
          <p className="text-xs text-muted-foreground">{deleteMessage}</p>
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
      <DeleteResourceModal
        isOpen={deleteOpen}
        title={t("customerPortal.hierarchy.deleteTitle", { kind })}
        alertMessage={t("customerPortal.hierarchy.deleteWarning")}
        message={deleteMessage}
        resourceInformationTitle={t("customerPortal.hierarchy.deleteInformationTitle", { kind: kindLabel })}
        resourceInformation={deleteInformation}
        requiredConfirmation={id}
        onCancel={() => setDeleteOpen(false)}
        onOk={() => void remove()}
        confirmLoading={pending === "delete"}
        errorMessage={deleteError}
      />
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
