import React, { useEffect, useState } from "react";
import { useWatch } from "react-hook-form";
import { apiClient } from "@/components/networking";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { AgentFormField, type AgentFormValues } from "./AgentFormKit";
import { entraTenantFromIssuer, IDENTITY_UUID_PATTERN } from "./agent_identity";

export const AgentIdentityFields = ({ accessToken }: { accessToken: string | null }) => {
  const provider = useWatch<AgentFormValues>({ name: "identity_provider" });
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
          <p className="text-sm text-muted-foreground">
            LiteLLM verifies the agent&apos;s Entra token before matching this identity. Saving these fields configures
            the binding; an authenticated request provides verification. Runtime authentication headers are configured
            separately.
          </p>
        </>
      )}
    </section>
  );
};
