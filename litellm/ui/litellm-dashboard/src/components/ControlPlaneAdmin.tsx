import { useTranslation } from "react-i18next";
import { useState, type FormEvent } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Plus } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import type { DataplanesQuery } from "@/contexts/ControlPlaneContext";
import {
  hasDataplane,
  hasScope,
  type ControlPlaneApi,
  type ControlPlaneDataplane,
  type ControlPlaneDataplaneInput,
  type ControlPlanePrincipal,
} from "@/lib/http/controlPlane";
import { ControlPlaneSectionHeader, ErrorNotice, ResultsTable, ScopedButton } from "./ControlPlaneShared";

type Role = Awaited<ReturnType<ControlPlaneApi["roles"]>>["items"][number];
type Section = "dataplanes" | "roles" | "audit";
type Editor = { action: "create" } | { action: "edit" | "delete"; resource: ControlPlaneDataplane | Role };

const read = (fields: FormData, name: string) => String(fields.get(name) ?? "").trim();
const list = (fields: FormData, name: string) =>
  read(fields, name)
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);

function dataplaneInput(fields: FormData): ControlPlaneDataplaneInput {
  const serviceToken = fields.get("serviceToken");
  const clientSecret = fields.get("clientSecret");
  if (typeof serviceToken !== "string" && typeof clientSecret !== "string")
    throw new Error("Configure at least one credential.");
  return {
    name: read(fields, "name") || null,
    url: read(fields, "url"),
    allow_insecure_http: fields.has("http"),
    service_token: typeof serviceToken === "string" ? { token: serviceToken } : null,
    oidc_exchange:
      typeof clientSecret === "string"
        ? {
            token_endpoint: read(fields, "tokenEndpoint"),
            client_id: read(fields, "clientId"),
            client_secret: clientSecret,
            audience: read(fields, "audience") || null,
            scope: read(fields, "scope") || null,
            profile: fields.get("profile") === "jwt_bearer_obo" ? "jwt_bearer_obo" : "rfc8693",
          }
        : null,
  };
}

function DataplaneFields({ plane }: { plane?: ControlPlaneDataplane }) {
  const { t } = useTranslation("translation", { keyPrefix: "controlPlane" });
  const [serviceToken, setServiceToken] = useState(plane?.credentials.includes("service_token") ?? true);
  const [oidc, setOidc] = useState(plane?.credentials.includes("oidc_exchange") ?? false);
  return (
    <>
      <label>
        {t("dataplaneId")}
        <Input name="id" defaultValue={plane?.id} readOnly={!!plane} required />
      </label>
      <label>
        {t("name")}
        <Input name="name" defaultValue={plane?.name} />
      </label>
      <label>
        {t("url")}
        <Input name="url" type="url" defaultValue={plane?.url} required />
      </label>
      <label className="flex gap-2">
        <input type="checkbox" name="http" defaultChecked={plane?.url.startsWith("http://")} /> {t("allowInsecureHttp")}
      </label>
      {plane && <p>{t("savingReplacesAllCredentialSettingsReenterCredentialsYouWantToRetain")}</p>}
      <label className="flex gap-2">
        <input type="checkbox" checked={serviceToken} onChange={(event) => setServiceToken(event.target.checked)} />{" "}
        {t("serviceToken")}
      </label>
      {serviceToken && (
        <label>
          {t("serviceTokenValue")}
          <Input name="serviceToken" type="password" autoComplete="new-password" required />
        </label>
      )}
      <label className="flex gap-2">
        <input type="checkbox" checked={oidc} onChange={(event) => setOidc(event.target.checked)} />{" "}
        {t("oidcTokenExchange")}
      </label>
      {oidc && (
        <>
          <label>
            {t("tokenEndpoint")}
            <Input name="tokenEndpoint" type="url" required />
          </label>
          <label>
            {t("clientId")}
            <Input name="clientId" required />
          </label>
          <label>
            {t("clientSecret")}
            <Input name="clientSecret" type="password" autoComplete="new-password" required />
          </label>
          <label>
            {t("audience")}
            <Input name="audience" />
          </label>
          <label>
            {t("scope")}
            <Input name="scope" />
          </label>
          <label className="grid gap-1">
            {t("profile")}
            <select className="rounded border bg-background px-3 py-2" name="profile">
              <option value="rfc8693">{t("rfc8693")}</option>
              <option value="jwt_bearer_obo">{t("jwtBearerOnbehalfof")}</option>
            </select>
          </label>
        </>
      )}
    </>
  );
}

function RegistryFields({ editor, section, id }: { editor: Editor; section: Exclude<Section, "audit">; id?: string }) {
  const { t } = useTranslation("translation", { keyPrefix: "controlPlane" });
  const resource = "resource" in editor ? editor.resource : undefined;
  if (editor.action === "delete") return <p>{t("confirmDeleteResource", { id })}</p>;
  if (section === "dataplanes") return <DataplaneFields plane={resource && "id" in resource ? resource : undefined} />;
  const role = resource && "scopes" in resource ? resource : undefined;
  const inherited = Boolean(role?.inherits.length);
  return (
    <>
      <label>
        {t("roleName")}
        <Input name="id" defaultValue={id} readOnly={!!id} required />
      </label>
      {inherited && <p>{t("replaceInheritedRole", { scopes: role?.scopes.join(", ") || t("none") })}</p>}
      <label>
        {t("directScopesCommaSeparated")}
        <Input name="scopes" defaultValue={inherited ? "" : role?.scopes.join(", ")} />
      </label>
      <label>
        {t("inheritedRolesCommaSeparated")}
        <Input name="inherits" defaultValue={role?.inherits.join(", ")} />
      </label>
      {inherited && (
        <label className="flex gap-2">
          <input required type="checkbox" /> {t("replaceTheExistingDirectScopesWithTheValuesEnteredAbove")}
        </label>
      )}
    </>
  );
}

function RegistryTable({
  items,
  section,
  actions,
  onRefresh,
  isFetching,
}: {
  items: readonly (ControlPlaneDataplane | Role)[];
  section: Exclude<Section, "audit">;
  actions: (item: ControlPlaneDataplane | Role) => React.ReactNode;
  onRefresh: () => void;
  isFetching: boolean;
}) {
  const { t } = useTranslation("translation", { keyPrefix: "controlPlane" });
  const columns =
    section === "dataplanes"
      ? [t("id"), t("name"), t("url"), t("credentials")]
      : [t("name"), t("effectiveScopes"), t("inheritedRoles")];
  const rows = items.map((item) => {
    const values =
      "id" in item
        ? [item.id, item.name, item.url, item.credentials.join(", ")]
        : [item.name, item.scopes.join(", "), item.inherits.join(", ")];
    return [...values, item.source === "config" ? t("configReadonly") : t("runtime"), actions(item)];
  });
  const rowKeys = items.map((item) => ("id" in item ? item.id : item.name));
  return (
    <ResultsTable
      columns={[...columns, t("source"), t("actions")]}
      rows={rows}
      rowKeys={rowKeys}
      onRefresh={onRefresh}
      isFetching={isFetching}
    />
  );
}

function Registry({
  api,
  planes,
  principal,
  section,
}: {
  api: ControlPlaneApi;
  planes: DataplanesQuery;
  principal: ControlPlanePrincipal;
  section: Exclude<Section, "audit">;
}) {
  const { t } = useTranslation("translation", { keyPrefix: "controlPlane" });
  const client = useQueryClient();
  const [editor, setEditor] = useState<Editor>();
  const [error, setError] = useState<unknown>();
  const [pending, setPending] = useState(false);
  const roles = useQuery({ queryKey: ["control-plane", "roles"], queryFn: api.roles, enabled: section === "roles" });
  const scope = section === "dataplanes" ? "dataplanes:write" : "roles:write";
  const reason = hasScope(principal, scope) ? undefined : t("requiresScope", { scope });
  const writeBlocked = reason ?? (principal.dataplanes !== null ? t("requiresFleetWide") : undefined);
  const resource = editor && "resource" in editor ? editor.resource : undefined;
  const label = section === "dataplanes" ? t("dataplaneResource") : t("roleResource");
  const result = section === "dataplanes" ? planes : roles;
  const items = result.data?.items.filter((item) => !("id" in item) || hasDataplane(principal, item.id));
  const id = resource && ("id" in resource ? resource.id : resource.name);
  const open = (value: Editor) => {
    setError(undefined);
    setEditor(value);
  };
  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (writeBlocked || pending || !editor) return;
    const fields = new FormData(event.currentTarget);
    const identifier = id ?? read(fields, "id");
    setPending(true);
    setError(undefined);
    try {
      if (editor.action === "create" && items?.some((item) => ("id" in item ? item.id : item.name) === identifier)) {
        throw new Error(t("idExists"));
      }
      if (editor.action === "delete") {
        if (section === "dataplanes") await api.deleteDataplane(identifier);
        else await api.deleteRole(identifier);
      } else if (section === "dataplanes") {
        await api.saveDataplane(identifier, dataplaneInput(fields));
      } else {
        await api.saveRole(identifier, { scopes: list(fields, "scopes"), inherits: list(fields, "inherits") });
      }
      setEditor(undefined);
      await Promise.all([client.invalidateQueries(), section === "dataplanes" ? planes.refetch() : undefined]);
    } catch (error) {
      setError(error);
    } finally {
      setPending(false);
    }
  }
  function actions(item: ControlPlaneDataplane | Role) {
    const disabled = item.source === "config" ? t("managedByConfig") : writeBlocked;
    return (
      <div className="flex gap-2">
        <ScopedButton
          variant="outline"
          reason={disabled}
          disabled={pending}
          onClick={() => open({ action: "edit", resource: item })}
        >
          {t("edit")}
        </ScopedButton>
        <ScopedButton
          variant="outline"
          reason={disabled}
          disabled={pending}
          onClick={() => open({ action: "delete", resource: item })}
        >
          {t("delete")}
        </ScopedButton>
      </div>
    );
  }
  return (
    <>
      <ControlPlaneSectionHeader
        section={section}
        action={
          <ScopedButton
            reason={writeBlocked ?? (!result.data ? t("loadTheRegistryBeforeCreatingAnEntry") : undefined)}
            disabled={pending}
            onClick={() => open({ action: "create" })}
          >
            <Plus /> {t("createResource", { label })}
          </ScopedButton>
        }
      />
      <ErrorNotice error={result.error} />
      <ErrorNotice error={error} />
      {result.isPending && <p role="status">{t("loadingRegistry", { label })}</p>}
      {editor && (
        <form key={`${editor.action}:${id ?? ""}`} className="grid max-w-lg gap-3 rounded border p-4" onSubmit={submit}>
          <h3 className="font-semibold">{t(`${editor.action}Resource`, { label })}</h3>
          <fieldset disabled={pending} className="grid gap-3">
            <RegistryFields editor={editor} section={section} id={id} />
          </fieldset>
          <div className="flex gap-2">
            <ScopedButton reason={writeBlocked} type="submit" disabled={pending}>
              {editor.action === "delete" ? t("confirmDelete") : t("save")}
            </ScopedButton>
            <Button onClick={() => setEditor(undefined)} variant="outline" disabled={pending}>
              {t("cancel")}
            </Button>
          </div>
        </form>
      )}
      {items && (
        <RegistryTable
          items={items}
          section={section}
          actions={actions}
          onRefresh={() => void result.refetch()}
          isFetching={result.isFetching}
        />
      )}
    </>
  );
}

function Audit({ api }: { api: ControlPlaneApi }) {
  const { t } = useTranslation("translation", { keyPrefix: "controlPlane" });
  const result = useQuery({ queryKey: ["control-plane", "audit"], queryFn: () => api.audit() });
  return (
    <>
      <ControlPlaneSectionHeader section="audit" />
      <ErrorNotice error={result.error} />
      {result.isPending && <p role="status">{t("loadingAuditLog")}</p>}
      {result.data && (
        <ResultsTable
          onRefresh={() => void result.refetch()}
          isFetching={result.isFetching}
          columns={[t("time"), t("actor"), t("dataplane"), t("action"), t("decision"), t("status"), t("detail")]}
          rows={result.data.items.map((event) => [
            event.at,
            event.actor,
            event.dataplane_id,
            `${event.method} ${event.path}`,
            event.decision,
            event.status,
            event.detail,
          ])}
        />
      )}
    </>
  );
}

export function ControlPlaneAdmin({
  api,
  planes,
  principal,
  section,
}: {
  api: ControlPlaneApi;
  planes: DataplanesQuery;
  principal: ControlPlanePrincipal;
  section: Section;
}) {
  if (section === "audit") return <Audit api={api} />;
  return <Registry api={api} principal={principal} planes={planes} section={section} />;
}
