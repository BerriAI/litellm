"use client";

import React from "react";
import { useFormContext, useWatch } from "react-hook-form";

import {
  FormControl,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from "@/components/ui/form";
import { Input } from "@/components/ui/input";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

import { ssoProviderDisplayNames, ssoProviderLogoMap } from "../constants";

export interface SSOSettingsFormValues {
  sso_provider?: string | null;
  google_client_id?: string | null;
  google_client_secret?: string | null;
  microsoft_client_id?: string | null;
  microsoft_client_secret?: string | null;
  microsoft_tenant?: string | null;
  generic_client_id?: string | null;
  generic_client_secret?: string | null;
  generic_authorization_endpoint?: string | null;
  generic_token_endpoint?: string | null;
  generic_userinfo_endpoint?: string | null;
  user_email?: string | null;
  proxy_base_url?: string | null;
  use_role_mappings?: boolean;
  group_claim?: string | null;
  default_role?: string | null;
  proxy_admin_teams?: string | null;
  admin_viewer_teams?: string | null;
  internal_user_teams?: string | null;
  internal_viewer_teams?: string | null;
  use_team_mappings?: boolean;
  team_ids_jwt_field?: string | null;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  [key: string]: any;
}

export interface BaseSSOSettingsFormProps {
  /**
   * Retained for API compatibility with previous antd-based callers. The
   * shadcn/RHF migration reads the form via `useFormContext`; callers now
   * wrap children in `FormProvider` and do not need to thread `form` down.
   */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  form?: any;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  onFormSubmit?: (formValues: Record<string, any>) => Promise<void>;
}

export interface SSOProviderConfig {
  envVarMap: Record<string, string>;
  fields: Array<{
    label: string;
    name: string;
    placeholder?: string;
  }>;
}

export const ssoProviderConfigs: Record<string, SSOProviderConfig> = {
  google: {
    envVarMap: {
      google_client_id: "GOOGLE_CLIENT_ID",
      google_client_secret: "GOOGLE_CLIENT_SECRET",
    },
    fields: [
      { label: "Google Client ID", name: "google_client_id" },
      { label: "Google Client Secret", name: "google_client_secret" },
    ],
  },
  microsoft: {
    envVarMap: {
      microsoft_client_id: "MICROSOFT_CLIENT_ID",
      microsoft_client_secret: "MICROSOFT_CLIENT_SECRET",
      microsoft_tenant: "MICROSOFT_TENANT",
    },
    fields: [
      { label: "Microsoft Client ID", name: "microsoft_client_id" },
      { label: "Microsoft Client Secret", name: "microsoft_client_secret" },
      { label: "Microsoft Tenant", name: "microsoft_tenant" },
    ],
  },
  okta: {
    envVarMap: {
      generic_client_id: "GENERIC_CLIENT_ID",
      generic_client_secret: "GENERIC_CLIENT_SECRET",
      generic_authorization_endpoint: "GENERIC_AUTHORIZATION_ENDPOINT",
      generic_token_endpoint: "GENERIC_TOKEN_ENDPOINT",
      generic_userinfo_endpoint: "GENERIC_USERINFO_ENDPOINT",
    },
    fields: [
      { label: "Generic Client ID", name: "generic_client_id" },
      { label: "Generic Client Secret", name: "generic_client_secret" },
      {
        label: "Authorization Endpoint",
        name: "generic_authorization_endpoint",
        placeholder: "https://your-domain/authorize",
      },
      {
        label: "Token Endpoint",
        name: "generic_token_endpoint",
        placeholder: "https://your-domain/token",
      },
      {
        label: "Userinfo Endpoint",
        name: "generic_userinfo_endpoint",
        placeholder: "https://your-domain/userinfo",
      },
    ],
  },
  generic: {
    envVarMap: {
      generic_client_id: "GENERIC_CLIENT_ID",
      generic_client_secret: "GENERIC_CLIENT_SECRET",
      generic_authorization_endpoint: "GENERIC_AUTHORIZATION_ENDPOINT",
      generic_token_endpoint: "GENERIC_TOKEN_ENDPOINT",
      generic_userinfo_endpoint: "GENERIC_USERINFO_ENDPOINT",
    },
    fields: [
      { label: "Generic Client ID", name: "generic_client_id" },
      { label: "Generic Client Secret", name: "generic_client_secret" },
      { label: "Authorization Endpoint", name: "generic_authorization_endpoint" },
      { label: "Token Endpoint", name: "generic_token_endpoint" },
      { label: "Userinfo Endpoint", name: "generic_userinfo_endpoint" },
    ],
  },
};

/**
 * Pure helper kept for call sites that render provider fields outside of
 * the full base form (e.g. `SSOModals.tsx`). The returned elements assume
 * an ambient `FormProvider` context, matching the parent form's RHF wiring.
 */
export const renderProviderFields = (provider: string) => {
  const config = ssoProviderConfigs[provider];
  if (!config) return null;

  return config.fields.map((field) => (
    <ProviderField
      key={field.name}
      label={field.label}
      name={field.name}
      placeholder={field.placeholder}
    />
  ));
};

const ProviderField: React.FC<{
  label: string;
  name: string;
  placeholder?: string;
}> = ({ label, name, placeholder }) => {
  const { control } = useFormContext();
  const isSecret = name.includes("client_secret");
  return (
    <FormField
      control={control}
      name={name}
      rules={{ required: `Please enter the ${label.toLowerCase()}` }}
      render={({ field }) => (
        <FormItem className="grid grid-cols-[8rem_1fr] items-start gap-4 space-y-0">
          <FormLabel className="pt-2">{label}</FormLabel>
          <div className="space-y-1">
            <FormControl>
              <Input
                type={isSecret ? "password" : "text"}
                placeholder={placeholder}
                {...field}
                value={field.value ?? ""}
              />
            </FormControl>
            <FormMessage />
          </div>
        </FormItem>
      )}
    />
  );
};

const BaseSSOSettingsForm: React.FC<BaseSSOSettingsFormProps> = () => {
  const { control } = useFormContext<SSOSettingsFormValues>();
  const provider = useWatch({ control, name: "sso_provider" });
  const useRoleMappings = useWatch({ control, name: "use_role_mappings" });
  const useTeamMappings = useWatch({ control, name: "use_team_mappings" });
  const supportsRoleMappings = provider === "okta" || provider === "generic";
  const supportsTeamMappings = provider === "okta" || provider === "generic";

  return (
    <div className="space-y-4">
      <FormField
        control={control}
        name="sso_provider"
        rules={{ required: "Please select an SSO provider" }}
        render={({ field }) => (
          <FormItem className="grid grid-cols-[8rem_1fr] items-start gap-4 space-y-0">
            <FormLabel className="pt-2">SSO Provider</FormLabel>
            <div className="space-y-1">
              <Select value={field.value ?? ""} onValueChange={field.onChange}>
                <FormControl>
                  <SelectTrigger>
                    <SelectValue placeholder="Select a provider" />
                  </SelectTrigger>
                </FormControl>
                <SelectContent>
                  {Object.entries(ssoProviderLogoMap).map(([value, logo]) => (
                    <SelectItem key={value} value={value}>
                      <div className="flex items-center py-1">
                        {logo && (
                          // eslint-disable-next-line @next/next/no-img-element
                          <img
                            src={logo}
                            alt={value}
                            className="h-6 w-6 mr-3 object-contain"
                          />
                        )}
                        <span>
                          {ssoProviderDisplayNames[value] ||
                            value.charAt(0).toUpperCase() + value.slice(1) + " SSO"}
                        </span>
                      </div>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <FormMessage />
            </div>
          </FormItem>
        )}
      />

      {provider ? renderProviderFields(provider) : null}

      <FormField
        control={control}
        name="user_email"
        rules={{ required: "Please enter the email of the proxy admin" }}
        render={({ field }) => (
          <FormItem className="grid grid-cols-[8rem_1fr] items-start gap-4 space-y-0">
            <FormLabel className="pt-2">Proxy Admin Email</FormLabel>
            <div className="space-y-1">
              <FormControl>
                <Input {...field} value={field.value ?? ""} />
              </FormControl>
              <FormMessage />
            </div>
          </FormItem>
        )}
      />

      <FormField
        control={control}
        name="proxy_base_url"
        rules={{
          required: "Please enter the proxy base url",
          validate: (value) => {
            const trimmed = (value ?? "").trim();
            if (!trimmed) return "Please enter the proxy base url";
            if (!/^https?:\/\/.+/.test(trimmed)) {
              return "URL must start with http:// or https://";
            }
            if (trimmed.endsWith("/")) {
              return "URL must not end with a trailing slash";
            }
            return true;
          },
        }}
        render={({ field }) => (
          <FormItem className="grid grid-cols-[8rem_1fr] items-start gap-4 space-y-0">
            <FormLabel className="pt-2">Proxy Base URL</FormLabel>
            <div className="space-y-1">
              <FormControl>
                <Input
                  placeholder="https://example.com"
                  {...field}
                  value={field.value ?? ""}
                  onBlur={(e) => {
                    field.onChange(e.target.value.trim());
                    field.onBlur();
                  }}
                />
              </FormControl>
              <FormMessage />
            </div>
          </FormItem>
        )}
      />

      {supportsRoleMappings && (
        <FormField
          control={control}
          name="use_role_mappings"
          render={({ field }) => (
            <FormItem className="grid grid-cols-[8rem_1fr] items-center gap-4 space-y-0">
              <FormLabel htmlFor="use_role_mappings">Use Role Mappings</FormLabel>
              <FormControl>
                <Checkbox
                  id="use_role_mappings"
                  checked={!!field.value}
                  onCheckedChange={(checked) => field.onChange(checked === true)}
                />
              </FormControl>
            </FormItem>
          )}
        />
      )}

      {useRoleMappings && supportsRoleMappings && (
        <>
          <FormField
            control={control}
            name="group_claim"
            rules={{ required: "Please enter the group claim" }}
            render={({ field }) => (
              <FormItem className="grid grid-cols-[8rem_1fr] items-start gap-4 space-y-0">
                <FormLabel className="pt-2">Group Claim</FormLabel>
                <div className="space-y-1">
                  <FormControl>
                    <Input {...field} value={field.value ?? ""} />
                  </FormControl>
                  <FormMessage />
                </div>
              </FormItem>
            )}
          />
          <FormField
            control={control}
            name="default_role"
            render={({ field }) => (
              <FormItem className="grid grid-cols-[8rem_1fr] items-start gap-4 space-y-0">
                <FormLabel className="pt-2">Default Role</FormLabel>
                <Select value={field.value ?? ""} onValueChange={field.onChange}>
                  <FormControl>
                    <SelectTrigger>
                      <SelectValue placeholder="Select a default role" />
                    </SelectTrigger>
                  </FormControl>
                  <SelectContent>
                    <SelectItem value="internal_user_viewer">Internal Viewer</SelectItem>
                    <SelectItem value="internal_user">Internal User</SelectItem>
                    <SelectItem value="proxy_admin_viewer">Admin Viewer</SelectItem>
                    <SelectItem value="proxy_admin">Proxy Admin</SelectItem>
                  </SelectContent>
                </Select>
              </FormItem>
            )}
          />
          <FormField
            control={control}
            name="proxy_admin_teams"
            render={({ field }) => (
              <FormItem className="grid grid-cols-[8rem_1fr] items-start gap-4 space-y-0">
                <FormLabel className="pt-2">Proxy Admin Teams</FormLabel>
                <FormControl>
                  <Input {...field} value={field.value ?? ""} />
                </FormControl>
              </FormItem>
            )}
          />
          <FormField
            control={control}
            name="admin_viewer_teams"
            render={({ field }) => (
              <FormItem className="grid grid-cols-[8rem_1fr] items-start gap-4 space-y-0">
                <FormLabel className="pt-2">Admin Viewer Teams</FormLabel>
                <FormControl>
                  <Input {...field} value={field.value ?? ""} />
                </FormControl>
              </FormItem>
            )}
          />
          <FormField
            control={control}
            name="internal_user_teams"
            render={({ field }) => (
              <FormItem className="grid grid-cols-[8rem_1fr] items-start gap-4 space-y-0">
                <FormLabel className="pt-2">Internal User Teams</FormLabel>
                <FormControl>
                  <Input {...field} value={field.value ?? ""} />
                </FormControl>
              </FormItem>
            )}
          />
          <FormField
            control={control}
            name="internal_viewer_teams"
            render={({ field }) => (
              <FormItem className="grid grid-cols-[8rem_1fr] items-start gap-4 space-y-0">
                <FormLabel className="pt-2">Internal Viewer Teams</FormLabel>
                <FormControl>
                  <Input {...field} value={field.value ?? ""} />
                </FormControl>
              </FormItem>
            )}
          />
        </>
      )}

      {supportsTeamMappings && (
        <FormField
          control={control}
          name="use_team_mappings"
          render={({ field }) => (
            <FormItem className="grid grid-cols-[8rem_1fr] items-center gap-4 space-y-0">
              <FormLabel htmlFor="use_team_mappings">Use Team Mappings</FormLabel>
              <FormControl>
                <Checkbox
                  id="use_team_mappings"
                  checked={!!field.value}
                  onCheckedChange={(checked) => field.onChange(checked === true)}
                />
              </FormControl>
            </FormItem>
          )}
        />
      )}

      {useTeamMappings && supportsTeamMappings && (
        <FormField
          control={control}
          name="team_ids_jwt_field"
          rules={{ required: "Please enter the team IDs JWT field" }}
          render={({ field }) => (
            <FormItem className="grid grid-cols-[8rem_1fr] items-start gap-4 space-y-0">
              <FormLabel className="pt-2">Team IDs JWT Field</FormLabel>
              <div className="space-y-1">
                <FormControl>
                  <Input {...field} value={field.value ?? ""} />
                </FormControl>
                <FormMessage />
              </div>
            </FormItem>
          )}
        />
      )}
    </div>
  );
};

export default BaseSSOSettingsForm;
