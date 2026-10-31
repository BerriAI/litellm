import React, { useEffect, useState } from "react";
import { useWatch } from "react-hook-form";
import type { ParseKeys } from "i18next";
import { Trans, useTranslation } from "react-i18next";
import i18n from "@/i18n/i18n";
import { apiClient } from "@/components/networking";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { AgentFormField, type AgentFormValues } from "./AgentFormKit";
import { entraTenantFromIssuer, IDENTITY_UUID_PATTERN } from "./agent_identity";

const PROVIDER_OPTIONS: { value: string; labelKey: ParseKeys }[] = [
  { value: "none", labelKey: "agents.identityFields.providers.none" },
  { value: "microsoft_entra", labelKey: "agents.identityFields.providers.microsoftEntra" },
];
const EXECUTION_MODE_OPTIONS: { value: string; labelKey: ParseKeys }[] = [
  { value: "autonomous", labelKey: "agents.identityFields.executionModes.autonomous" },
  { value: "delegated", labelKey: "agents.identityFields.executionModes.delegated" },
  { value: "both", labelKey: "agents.identityFields.executionModes.both" },
];
const EXECUTION_OPTIONS: { value: string; labelKey: ParseKeys }[] = [
  { value: "enabled", labelKey: "agents.identityFields.execution.enabled" },
  { value: "disabled", labelKey: "agents.identityFields.execution.disabled" },
];

export const AgentIdentityFields = ({ accessToken }: { accessToken: string | null }) => {
  const { t } = useTranslation();
  const provider = useWatch<AgentFormValues>({ name: "identity_provider" });
  const mode = useWatch<AgentFormValues>({ name: "execution_mode" });
  const showScopes = mode !== "autonomous" && mode !== undefined;
  const [tenants, setTenants] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!accessToken || provider !== "microsoft_entra") return;
    let active = true;
    apiClient
      .get<string[]>("/v1/agents/identity/providers", { accessToken })
      .then((issuers) => {
        if (active) {
          setError(null);
          setTenants(
            issuers.flatMap((issuer) => {
              const tenant = entraTenantFromIssuer(issuer);
              return tenant ? [tenant] : [];
            }),
          );
        }
      })
      .catch(() => {
        if (active) setError(i18n.t("agents.identityFields.errors.providersLoadFailed"));
      });
    return () => {
      active = false;
    };
  }, [accessToken, provider]);

  return (
    <>
      <section
        aria-label={t("agents.identityFields.regionLabel")}
        className="my-6 space-y-4 rounded-lg border border-border p-4"
      >
        <div>
          <h3 className="font-medium">{t("agents.identityFields.title")}</h3>
          <p className="mt-1 text-sm text-muted-foreground">{t("agents.identityFields.description")}</p>
        </div>
        <AgentFormField name="identity_provider" label={t("agents.identityFields.provider")} defaultValue="none">
          {({ value, onChange, id }) => (
            <Select
              items={PROVIDER_OPTIONS.map((option) => ({ value: option.value, label: t(option.labelKey) }))}
              value={typeof value === "string" ? value : "none"}
              onValueChange={onChange}
            >
              <SelectTrigger id={id}>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {PROVIDER_OPTIONS.map((option) => (
                  <SelectItem key={option.value} value={option.value}>
                    {t(option.labelKey)}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          )}
        </AgentFormField>
        {provider === "microsoft_entra" && (
          <>
            <AgentFormField
              name="identity_tenant_id"
              label={t("agents.identityFields.tenant.label")}
              rules={{ required: t("agents.identityFields.tenant.required") }}
            >
              {({ value, onChange, id }) => (
                <Select value={typeof value === "string" ? value : ""} onValueChange={onChange}>
                  <SelectTrigger id={id}>
                    <SelectValue placeholder={t("agents.identityFields.tenant.placeholder")} />
                  </SelectTrigger>
                  <SelectContent>
                    {tenants.map((tenant) => (
                      <SelectItem key={tenant} value={tenant}>
                        {tenant}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              )}
            </AgentFormField>
            {error && (
              <p role="alert" className="text-sm text-destructive">
                {error}
              </p>
            )}
            {!error && tenants.length === 0 && (
              <p className="text-sm text-muted-foreground">{t("agents.identityFields.tenant.noneAvailable")}</p>
            )}
            <AgentFormField
              name="identity_client_id"
              label={t("agents.identityFields.clientId.label")}
              rules={{
                required: t("agents.identityFields.clientId.required"),
                pattern: { value: IDENTITY_UUID_PATTERN, message: t("agents.identityFields.clientId.invalid") },
              }}
              description={
                <Trans
                  i18nKey="agents.identityFields.clientId.description"
                  components={{
                    registrations: (
                      <a className="underline" href="https://entra.microsoft.com/" target="_blank" rel="noreferrer" />
                    ),
                  }}
                />
              }
            >
              {({ value, onChange, ref, ...control }) => (
                <Input
                  {...control}
                  ref={ref}
                  value={typeof value === "string" ? value : ""}
                  onChange={onChange}
                  placeholder="xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
                />
              )}
            </AgentFormField>
            <AgentFormField
              name="execution_mode"
              label={t("agents.identityFields.executionMode")}
              defaultValue="autonomous"
            >
              {({ value, onChange, id }) => (
                <Select
                  items={EXECUTION_MODE_OPTIONS.map((option) => ({ value: option.value, label: t(option.labelKey) }))}
                  value={typeof value === "string" ? value : "autonomous"}
                  onValueChange={onChange}
                >
                  <SelectTrigger id={id}>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {EXECUTION_MODE_OPTIONS.map((option) => (
                      <SelectItem key={option.value} value={option.value}>
                        {t(option.labelKey)}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              )}
            </AgentFormField>
            <AgentFormField
              name="identity_service_principal_id"
              label={t("agents.identityFields.servicePrincipal.label")}
              rules={{
                required: mode !== "delegated" ? t("agents.identityFields.servicePrincipal.required") : false,
                pattern: { value: IDENTITY_UUID_PATTERN, message: t("agents.identityFields.servicePrincipal.invalid") },
              }}
              description={
                <Trans
                  i18nKey="agents.identityFields.servicePrincipal.description"
                  components={{
                    enterprise: (
                      <a
                        className="underline"
                        href="https://entra.microsoft.com/#view/Microsoft_AAD_IAM/StartboardApplicationsMenuBlade/~/AppAppsPreview"
                        target="_blank"
                        rel="noreferrer"
                      />
                    ),
                  }}
                />
              }
            >
              {({ value, onChange, ref, ...control }) => (
                <Input {...control} ref={ref} value={typeof value === "string" ? value : ""} onChange={onChange} />
              )}
            </AgentFormField>

            <AgentFormField
              name="identity_required_roles"
              label={t("agents.identityFields.requiredRoles.label")}
              description={t("agents.identityFields.requiredRoles.description")}
            >
              {({ value, onChange, ref, ...control }) => (
                <Input
                  {...control}
                  ref={ref}
                  value={typeof value === "string" ? value : ""}
                  onChange={onChange}
                  placeholder="Agent.Invoke"
                />
              )}
            </AgentFormField>

            {showScopes && (
              <>
                <AgentFormField
                  name="identity_required_scopes"
                  label={t("agents.identityFields.requiredScopes.label")}
                  defaultValue="user_impersonation"
                  rules={{ required: t("agents.identityFields.requiredScopes.required") }}
                >
                  {({ value, onChange, ref, ...control }) => (
                    <Input
                      {...control}
                      ref={ref}
                      value={typeof value === "string" ? value : "user_impersonation"}
                      onChange={onChange}
                    />
                  )}
                </AgentFormField>
                <p className="text-sm text-muted-foreground">{t("agents.identityFields.requiredScopes.note")}</p>
              </>
            )}
            <AgentFormField name="enabled" label={t("agents.identityFields.executionLabel")} defaultValue={true}>
              {({ value, onChange, id }) => (
                <Select
                  items={EXECUTION_OPTIONS.map((option) => ({ value: option.value, label: t(option.labelKey) }))}
                  value={value === false ? "disabled" : "enabled"}
                  onValueChange={(next) => onChange(next === "enabled")}
                >
                  <SelectTrigger id={id}>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {EXECUTION_OPTIONS.map((option) => (
                      <SelectItem key={option.value} value={option.value}>
                        {t(option.labelKey)}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              )}
            </AgentFormField>
            <p className="text-sm text-muted-foreground">{t("agents.identityFields.verificationNote")}</p>
          </>
        )}
      </section>
    </>
  );
};
