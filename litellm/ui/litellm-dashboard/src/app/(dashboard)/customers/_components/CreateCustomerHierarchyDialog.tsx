"use client";

import { useState, type FormEvent } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { toast } from "sonner";
import { fetchClient } from "@/lib/http/api";
import { extractProxyErrorMessage } from "@/lib/http/client";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import CustomerHierarchyBudgetFields from "./CustomerHierarchyBudgetFields";
import {
  buildCustomerHierarchyCreate,
  newCustomerHierarchyFields,
  CUSTOMER_HIERARCHY_FORM_ERRORS,
  type CreatedCustomerHierarchyEntity,
  type CustomerHierarchyCreate,
  type CustomerHierarchyCreateTarget,
  type CustomerHierarchyFields,
} from "./customerHierarchyForms";

type Props = CustomerHierarchyCreateTarget & {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onCreated: (entity: CreatedCustomerHierarchyEntity) => void;
};

const createEntity = async (create: CustomerHierarchyCreate): Promise<CreatedCustomerHierarchyEntity | null> => {
  if (create.kind === "team") {
    const result = await fetchClient.POST("/management/v1/customer_teams", { body: create.body });
    if (result.response.status !== 201 || !result.data) return null;
    return { kind: "team", record: result.data };
  }
  const result = await fetchClient.POST("/management/v1/customer_members", { body: create.body });
  if (result.response.status !== 201 || !result.data) return null;
  return { kind: "member", record: result.data };
};

function CreateForm(props: Props & { pending: boolean; onPendingChange: (pending: boolean) => void }) {
  const queryClient = useQueryClient();
  const { t } = useTranslation();
  const [fields, setFields] = useState(() => newCustomerHierarchyFields(props));
  const [identifier, setIdentifier] = useState("");
  const { pending, onPendingChange: setPending } = props;
  const [error, setError] = useState<string | null>(null);
  const update = <K extends keyof CustomerHierarchyFields>(key: K, value: CustomerHierarchyFields[K]) =>
    setFields((current) => ({ ...current, [key]: value }));
  const title = t(
    props.kind === "team" ? "customerPortal.hierarchy.createTeam" : "customerPortal.hierarchy.createMember",
  );
  const hint =
    props.kind === "team"
      ? t("customerPortal.hierarchy.createTeamHint", {
          organization: props.organization.alias || props.organization.customer_org_id,
        })
      : t("customerPortal.hierarchy.createMemberHint", { team: props.team.alias || props.team.customer_team_id });

  const submit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const result = buildCustomerHierarchyCreate(props, fields, identifier);
    if (!result.ok) {
      setError(t(CUSTOMER_HIERARCHY_FORM_ERRORS[result.error]));
      return;
    }
    setPending(true);
    setError(null);
    try {
      const created = await createEntity(result.create);
      if (!created) {
        setError(t("customerPortal.hierarchy.createFailed"));
        return;
      }
      toast.success(
        t("customerPortal.hierarchy.created", {
          kind: t(props.kind === "team" ? "customerPortal.hierarchy.team" : "customerPortal.hierarchy.member"),
        }),
      );
      props.onOpenChange(false);
      props.onCreated(created);
    } catch (cause) {
      setError(extractProxyErrorMessage(cause));
    } finally {
      void queryClient.invalidateQueries({ queryKey: ["get", "/management/v1/customer_portal/limits"] });
      setPending(false);
    }
  };

  return (
    <DialogContent className="max-h-[85vh] overflow-y-auto sm:max-w-2xl" showCloseButton={!pending}>
      <form className="space-y-5" onSubmit={(event) => void submit(event)}>
        <DialogHeader>
          <DialogTitle>{title}</DialogTitle>
          <DialogDescription>{hint}</DialogDescription>
        </DialogHeader>
        <fieldset disabled={pending} className="space-y-5">
          <div className="grid gap-4 sm:grid-cols-2">
            <div className="space-y-1.5">
              <Label htmlFor="create-hierarchy-alias">{t("customerPortal.hierarchy.alias")}</Label>
              <Input
                id="create-hierarchy-alias"
                value={fields.alias}
                onChange={(event) => update("alias", event.target.value)}
                autoFocus
              />
            </div>
            <div className="space-y-1.5">
              <Label htmlFor="create-hierarchy-id">{t("customerPortal.hierarchy.customIdentifier")}</Label>
              <Input
                id="create-hierarchy-id"
                value={identifier}
                onChange={(event) => setIdentifier(event.target.value)}
                maxLength={256}
                aria-describedby="create-hierarchy-id-hint"
              />
              <p className="text-xs text-muted-foreground" id="create-hierarchy-id-hint">
                {t("customerPortal.hierarchy.identifierAutoGenerated")}
              </p>
            </div>
          </div>
          <CustomerHierarchyBudgetFields
            kind={props.kind}
            organization={props.organization}
            fields={fields}
            onChange={update}
            idPrefix="create-hierarchy"
            operation="create"
          />
        </fieldset>
        {error && (
          <p role="alert" className="text-sm text-destructive">
            {error}
          </p>
        )}
        <DialogFooter>
          <Button type="button" variant="outline" disabled={pending} onClick={() => props.onOpenChange(false)}>
            {t("customerPortal.hierarchy.cancel")}
          </Button>
          <Button type="submit" disabled={pending}>
            {t(pending ? "customerPortal.hierarchy.creating" : "customerPortal.hierarchy.create")}
          </Button>
        </DialogFooter>
      </form>
    </DialogContent>
  );
}

export default function CreateCustomerHierarchyDialog(props: Props) {
  const [pending, setPending] = useState(false);
  const parent = props.kind === "team" ? props.organization.customer_org_id : props.team.customer_team_id;
  return (
    <Dialog
      open={props.open}
      onOpenChange={(open) => {
        if (!pending) props.onOpenChange(open);
      }}
    >
      {props.open && (
        <CreateForm key={`${props.kind}:${parent}`} {...props} pending={pending} onPendingChange={setPending} />
      )}
    </Dialog>
  );
}
