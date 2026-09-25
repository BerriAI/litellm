import React, { useEffect, useState } from "react";
import { useWatch } from "react-hook-form";
import { apiClient } from "@/components/networking";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { AgentFormField, type AgentFormValues } from "./AgentFormKit";
import { entraTenantFromIssuer, IDENTITY_UUID_PATTERN } from "./agent_identity";

export const AgentIdentityFields = ({ accessToken }: { accessToken: string | null }) => {
  const provider = useWatch<AgentFormValues>({ name: "identity_provider" });
  const mode = useWatch<AgentFormValues>({ name: "execution_mode" });
  const [tenants, setTenants] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!accessToken || provider !== "microsoft_entra") return;
    let active = true;
    apiClient
      .get<string[]>("/v1/agents/identity/providers", { accessToken })
      .then((issuers) => {
        if (active)
          setTenants(
            issuers.flatMap((issuer) => {
              const tenant = entraTenantFromIssuer(issuer);
              return tenant ? [tenant] : [];
            }),
          );
      })
      .catch(() => {
        if (active) setError("Could not load the gateway's trusted identity providers");
      });
    return () => {
      active = false;
    };
  }, [accessToken, provider]);

  return (
    <>
      <section aria-label="Agent Identity" className="my-6 space-y-4 rounded-lg border border-border p-4">
        <div>
          <h3 className="font-medium">Agent Identity</h3>
          <p className="mt-1 text-sm text-muted-foreground">
            Connect an existing identity provider application to this agent. Its name and runtime address can change
            independently.
          </p>
        </div>
        <AgentFormField name="identity_provider" label="Identity Provider" defaultValue="none">
          {({ value, onChange, id }) => (
            <Select value={typeof value === "string" ? value : "none"} onValueChange={onChange}>
              <SelectTrigger id={id}>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="none">No explicit identity binding</SelectItem>
                <SelectItem value="microsoft_entra">Microsoft Entra ID</SelectItem>
              </SelectContent>
            </Select>
          )}
        </AgentFormField>
        {provider === "microsoft_entra" && (
          <>
            <AgentFormField
              name="identity_tenant_id"
              label="Trusted Entra Tenant"
              rules={{ required: "Select a trusted tenant" }}
            >
              {({ value, onChange, id }) => (
                <Select value={typeof value === "string" ? value : ""} onValueChange={onChange}>
                  <SelectTrigger id={id}>
                    <SelectValue placeholder="Select the gateway's trusted tenant" />
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
              <p className="text-sm text-muted-foreground">
                No trusted Entra tenant is available. Configure JWT issuer and audience validation on the gateway first.
                Dashboard Microsoft SSO is configured separately.
              </p>
            )}
            <AgentFormField
              name="identity_client_id"
              label="Application (Client) ID"
              rules={{
                required: "Enter the Entra application client ID",
                pattern: { value: IDENTITY_UUID_PATTERN, message: "Enter a valid application client UUID" },
              }}
              description={
                <>
                  Find this under{" "}
                  <a className="underline" href="https://entra.microsoft.com/" target="_blank" rel="noreferrer">
                    Entra App registrations
                  </a>
                  , select your agent application, then Overview. No client secret is required here.
                </>
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
            <AgentFormField name="execution_mode" label="Execution Mode" defaultValue="autonomous">
              {({ value, onChange, id }) => (
                <Select value={typeof value === "string" ? value : "autonomous"} onValueChange={onChange}>
                  <SelectTrigger id={id}>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="autonomous">Autonomous</SelectItem>
                    <SelectItem value="delegated">On behalf of a user</SelectItem>
                    <SelectItem value="both">Both</SelectItem>
                  </SelectContent>
                </Select>
              )}
            </AgentFormField>
            <AgentFormField
              name="identity_service_principal_id"
              label="Enterprise Application Object ID"
              rules={{
                required: mode !== "delegated" ? "Enter the service principal Object ID" : false,
                pattern: { value: IDENTITY_UUID_PATTERN, message: "Enter a valid service principal UUID" },
              }}
              description={
                <>
                  Open{" "}
                  <a
                    className="underline"
                    href="https://entra.microsoft.com/#view/Microsoft_AAD_IAM/StartboardApplicationsMenuBlade/~/AppAppsPreview"
                    target="_blank"
                    rel="noreferrer"
                  >
                    Entra Enterprise applications
                  </a>
                  , select this application, and copy its Object ID. The App registrations Object ID is a different
                  value.
                </>
              }
            >
              {({ value, onChange, ref, ...control }) => (
                <Input {...control} ref={ref} value={typeof value === "string" ? value : ""} onChange={onChange} />
              )}
            </AgentFormField>
            <AgentFormField
              name="identity_required_roles"
              label="Required Application Roles"
              description="Comma-separated role values required on autonomous application tokens"
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
            {mode !== "autonomous" && mode !== undefined && (
              <>
                <AgentFormField
                  name="identity_required_scopes"
                  label="Required Delegated Scopes"
                  defaultValue="user_impersonation"
                  rules={{ required: "Enter a delegated scope" }}
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
                <p className="text-sm text-muted-foreground">
                  Users must first sign in through this gateway&apos;s Microsoft SSO. Subsequent delegated calls must
                  satisfy both user and agent permissions.
                </p>
              </>
            )}
            <AgentFormField name="enabled" label="Execution" defaultValue={true}>
              {({ value, onChange, id }) => (
                <Select
                  value={value === false ? "disabled" : "enabled"}
                  onValueChange={(next) => onChange(next === "enabled")}
                >
                  <SelectTrigger id={id}>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="enabled">Enabled</SelectItem>
                    <SelectItem value="disabled">Disabled</SelectItem>
                  </SelectContent>
                </Select>
              )}
            </AgentFormField>
            <p className="text-sm text-muted-foreground">
              LiteLLM verifies the agent&apos;s Entra token before matching this identity. Saving these fields
              configures the binding; an authenticated request provides verification. Runtime authentication headers are
              configured separately.
            </p>
          </>
        )}
      </section>
      <section aria-label="Agent Budget" className="my-6 space-y-4 rounded-lg border border-border p-4">
        <h3 className="font-medium">Agent Budget</h3>
        <AgentFormField
          name="agent_max_budget"
          label="Aggregate Agent Budget ($)"
          description="Shared across this agent's requests and keys. Leave empty for no aggregate limit"
        >
          {({ value, onChange, ref, ...control }) => (
            <Input
              {...control}
              ref={ref}
              type="number"
              min="0"
              step="any"
              value={typeof value === "number" || typeof value === "string" ? value : ""}
              onChange={onChange}
            />
          )}
        </AgentFormField>
        <AgentFormField
          name="agent_budget_duration"
          label="Budget Reset Period"
          description="For example, 1d or 30d. Leave empty for a lifetime budget"
        >
          {({ value, onChange, ref, ...control }) => (
            <Input
              {...control}
              ref={ref}
              value={typeof value === "string" ? value : ""}
              onChange={onChange}
              placeholder="30d"
            />
          )}
        </AgentFormField>
      </section>
    </>
  );
};
