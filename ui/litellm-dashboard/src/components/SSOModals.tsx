import React, { useEffect, useState } from "react";
import { Controller, FormProvider, useForm, useFormContext } from "react-hook-form";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { getSSOSettings, updateSSOSettings } from "./networking";
import NotificationsManager from "./molecules/notifications_manager";
import { parseErrorMessage } from "./shared/errorUtils";

interface SSOModalsProps {
  isAddSSOModalVisible: boolean;
  isInstructionsModalVisible: boolean;
  handleAddSSOOk: () => void;
  handleAddSSOCancel: () => void;
  handleShowInstructions: (formValues: Record<string, any>) => void;
  handleInstructionsOk: () => void;
  handleInstructionsCancel: () => void;
  /**
   * Optional external form handle. For backwards compatibility callers may pass
   * an antd `FormInstance` (from `Form.useForm()`); we adapt its common
   * methods (`resetFields`, `setFieldsValue`, `getFieldsValue`) to the
   * internal `react-hook-form` state. New callers may omit this entirely.
   */
  form?: any;
  accessToken: string | null;
  ssoConfigured?: boolean;
}

const ssoProviderLogoMap: Record<string, string> = {
  google: "https://artificialanalysis.ai/img/logos/google_small.svg",
  microsoft: "https://upload.wikimedia.org/wikipedia/commons/a/a8/Microsoft_Azure_Logo.svg",
  okta: "https://www.okta.com/sites/default/files/Okta_Logo_BrightBlue_Medium.png",
  generic: "",
};

interface SSOProviderConfig {
  envVarMap: Record<string, string>;
  fields: Array<{
    label: string;
    name: string;
    placeholder?: string;
  }>;
}

const ssoProviderConfigs: Record<string, SSOProviderConfig> = {
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
      { label: "Token Endpoint", name: "generic_token_endpoint", placeholder: "https://your-domain/token" },
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

const providerFieldKeys: string[] = Object.values(ssoProviderConfigs).flatMap(
  (cfg) => cfg.fields.map((f) => f.name),
);

interface SSOFormValues {
  sso_provider: string;
  proxy_base_url: string;
  user_email: string;
  use_role_mappings: boolean;
  group_claim: string;
  default_role: string;
  proxy_admin_teams: string;
  admin_viewer_teams: string;
  internal_user_teams: string;
  internal_viewer_teams: string;
  [key: string]: any;
}

const defaultFormValues: SSOFormValues = {
  sso_provider: "",
  proxy_base_url: "",
  user_email: "",
  use_role_mappings: false,
  group_claim: "",
  default_role: "internal_user",
  proxy_admin_teams: "",
  admin_viewer_teams: "",
  internal_user_teams: "",
  internal_viewer_teams: "",
  google_client_id: "",
  google_client_secret: "",
  microsoft_client_id: "",
  microsoft_client_secret: "",
  microsoft_tenant: "",
  generic_client_id: "",
  generic_client_secret: "",
  generic_authorization_endpoint: "",
  generic_token_endpoint: "",
  generic_userinfo_endpoint: "",
};

const URL_PATTERN = /^https?:\/\/.+/;

function ProviderFields({ provider }: { provider: string }) {
  const { control } = useFormContext<SSOFormValues>();
  const config = ssoProviderConfigs[provider];
  if (!config) return null;
  return (
    <>
      {config.fields.map((field) => {
        const isClient = field.name.includes("client");
        return (
          <Controller
            key={field.name}
            control={control}
            name={field.name as any}
            rules={{ required: `Please enter the ${field.label.toLowerCase()}` }}
            render={({ field: rhfField, fieldState }) => (
              <div className="grid grid-cols-[1fr_2fr] gap-4 items-start">
                <Label htmlFor={`sso-${field.name}`} className="mt-2 text-left">
                  {field.label} <span className="text-destructive">*</span>
                </Label>
                <div className="space-y-1">
                  <Input
                    id={`sso-${field.name}`}
                    type={isClient ? "password" : "text"}
                    placeholder={field.placeholder}
                    value={(rhfField.value as string) ?? ""}
                    onChange={rhfField.onChange}
                    onBlur={rhfField.onBlur}
                    name={rhfField.name}
                    ref={rhfField.ref}
                  />
                  {fieldState.error && (
                    <p className="text-sm text-destructive">{fieldState.error.message}</p>
                  )}
                </div>
              </div>
            )}
          />
        );
      })}
    </>
  );
}

function RoleMappingFields() {
  const { control, register } = useFormContext<SSOFormValues>();
  return (
    <>
      <Controller
        control={control}
        name="group_claim"
        rules={{ required: "Please enter the group claim" }}
        render={({ field, fieldState }) => (
          <div className="grid grid-cols-[1fr_2fr] gap-4 items-start">
            <Label htmlFor="sso-group-claim" className="mt-2 text-left">
              Group Claim <span className="text-destructive">*</span>
            </Label>
            <div className="space-y-1">
              <Input id="sso-group-claim" value={field.value ?? ""} onChange={field.onChange} />
              {fieldState.error && (
                <p className="text-sm text-destructive">{fieldState.error.message}</p>
              )}
            </div>
          </div>
        )}
      />
      <Controller
        control={control}
        name="default_role"
        render={({ field }) => (
          <div className="grid grid-cols-[1fr_2fr] gap-4 items-center">
            <Label htmlFor="sso-default-role" className="text-left">
              Default Role
            </Label>
            <Select value={(field.value as string) || "internal_user"} onValueChange={field.onChange}>
              <SelectTrigger id="sso-default-role">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="internal_user_viewer">Internal Viewer</SelectItem>
                <SelectItem value="internal_user">Internal User</SelectItem>
                <SelectItem value="proxy_admin_viewer">Admin Viewer</SelectItem>
                <SelectItem value="proxy_admin">Proxy Admin</SelectItem>
              </SelectContent>
            </Select>
          </div>
        )}
      />
      {[
        { name: "proxy_admin_teams", label: "Proxy Admin Teams" },
        { name: "admin_viewer_teams", label: "Admin Viewer Teams" },
        { name: "internal_user_teams", label: "Internal User Teams" },
        { name: "internal_viewer_teams", label: "Internal Viewer Teams" },
      ].map((f) => (
        <div key={f.name} className="grid grid-cols-[1fr_2fr] gap-4 items-center">
          <Label htmlFor={`sso-${f.name}`} className="text-left">
            {f.label}
          </Label>
          <Input id={`sso-${f.name}`} {...register(f.name as any)} />
        </div>
      ))}
    </>
  );
}

const SSOModals: React.FC<SSOModalsProps> = ({
  isAddSSOModalVisible,
  isInstructionsModalVisible,
  handleAddSSOOk,
  handleAddSSOCancel,
  handleShowInstructions,
  handleInstructionsOk,
  handleInstructionsCancel,
  form: externalForm,
  accessToken,
  ssoConfigured = false,
}) => {
  const [isClearConfirmModalVisible, setIsClearConfirmModalVisible] = useState(false);
  const rhfForm = useForm<SSOFormValues>({
    defaultValues: defaultFormValues,
    mode: "onSubmit",
  });

  // Bridge the optional antd-style `form` prop to the internal RHF form, so
  // existing callers that still do `Form.useForm()` and call `resetFields` /
  // `setFieldsValue` on the instance keep working during the migration.
  useEffect(() => {
    if (!externalForm) return;
    const patchedResetFields = () => rhfForm.reset(defaultFormValues);
    const patchedSetFieldsValue = (values: Partial<SSOFormValues>) => {
      Object.entries(values).forEach(([k, v]) => {
        rhfForm.setValue(k as any, v as any, { shouldDirty: false });
      });
    };
    const patchedGetFieldsValue = () => rhfForm.getValues();
    externalForm.resetFields = patchedResetFields;
    externalForm.setFieldsValue = patchedSetFieldsValue;
    externalForm.getFieldsValue = patchedGetFieldsValue;
  }, [externalForm, rhfForm]);

  const provider = rhfForm.watch("sso_provider");
  const useRoleMappings = rhfForm.watch("use_role_mappings");

  useEffect(() => {
    const loadSSOSettings = async () => {
      if (isAddSSOModalVisible && accessToken) {
        try {
          const ssoData = await getSSOSettings(accessToken);
          if (ssoData && ssoData.values) {
            let selectedProvider: string = "";
            if (ssoData.values.google_client_id) {
              selectedProvider = "google";
            } else if (ssoData.values.microsoft_client_id) {
              selectedProvider = "microsoft";
            } else if (ssoData.values.generic_client_id) {
              if (
                ssoData.values.generic_authorization_endpoint?.includes("okta") ||
                ssoData.values.generic_authorization_endpoint?.includes("auth0")
              ) {
                selectedProvider = "okta";
              } else {
                selectedProvider = "generic";
              }
            }

            let roleMappingFields: Partial<SSOFormValues> = {};
            if (ssoData.values.role_mappings) {
              const roleMappings = ssoData.values.role_mappings;

              const joinTeams = (teams: string[] | undefined): string => {
                if (!teams || teams.length === 0) return "";
                return teams.join(", ");
              };

              roleMappingFields = {
                use_role_mappings: true,
                group_claim: roleMappings.group_claim,
                default_role: roleMappings.default_role || "internal_user",
                proxy_admin_teams: joinTeams(roleMappings.roles?.proxy_admin),
                admin_viewer_teams: joinTeams(roleMappings.roles?.proxy_admin_viewer),
                internal_user_teams: joinTeams(roleMappings.roles?.internal_user),
                internal_viewer_teams: joinTeams(roleMappings.roles?.internal_user_viewer),
              };
            }

            const formValues = {
              ...defaultFormValues,
              sso_provider: selectedProvider,
              proxy_base_url: ssoData.values.proxy_base_url ?? "",
              user_email: ssoData.values.user_email ?? "",
              ...ssoData.values,
              ...roleMappingFields,
            };

            rhfForm.reset({
              ...defaultFormValues,
              ...formValues,
            });
          }
        } catch (error) {
          console.error("Failed to load SSO settings:", error);
        }
      }
    };

    loadSSOSettings();
  }, [isAddSSOModalVisible, accessToken, rhfForm]);

  const onSubmit = rhfForm.handleSubmit(async (formValues) => {
    if (!accessToken) {
      NotificationsManager.fromBackend("No access token available");
      return;
    }

    try {
      const {
        proxy_admin_teams,
        admin_viewer_teams,
        internal_user_teams,
        internal_viewer_teams,
        default_role,
        group_claim,
        use_role_mappings,
        ...rest
      } = formValues;

      // Strip fields that belong to providers other than the selected one, and
      // drop any empty strings so we don't send blank fields to the backend.
      const selectedConfig = ssoProviderConfigs[rest.sso_provider];
      const allowedProviderKeys = new Set(
        selectedConfig ? selectedConfig.fields.map((f) => f.name) : [],
      );
      const payload: Record<string, any> = {
        sso_provider: rest.sso_provider,
        user_email: rest.user_email,
        proxy_base_url: rest.proxy_base_url,
      };
      for (const key of providerFieldKeys) {
        if (allowedProviderKeys.has(key)) {
          const v = (rest as any)[key];
          if (v !== undefined && v !== null && v !== "") {
            payload[key] = v;
          }
        }
      }

      if (use_role_mappings) {
        const splitTeams = (teams: string | undefined): string[] => {
          if (!teams || teams.trim() === "") return [];
          return teams
            .split(",")
            .map((team) => team.trim())
            .filter((team) => team.length > 0);
        };

        const defaultRoleMapping: Record<string, string> = {
          internal_user_viewer: "internal_user_viewer",
          internal_user: "internal_user",
          proxy_admin_viewer: "proxy_admin_viewer",
          proxy_admin: "proxy_admin",
        };

        payload.role_mappings = {
          provider: "generic",
          group_claim,
          default_role: defaultRoleMapping[default_role] || "internal_user",
          roles: {
            proxy_admin: splitTeams(proxy_admin_teams),
            proxy_admin_viewer: splitTeams(admin_viewer_teams),
            internal_user: splitTeams(internal_user_teams),
            internal_user_viewer: splitTeams(internal_viewer_teams),
          },
        };
      }

      await updateSSOSettings(accessToken, payload);

      handleShowInstructions(formValues);
    } catch (error: unknown) {
      NotificationsManager.fromBackend("Failed to save SSO settings: " + parseErrorMessage(error));
    }
  });

  const handleClearSSO = async () => {
    if (!accessToken) {
      NotificationsManager.fromBackend("No access token available");
      return;
    }

    try {
      const clearSettings = {
        google_client_id: null,
        google_client_secret: null,
        microsoft_client_id: null,
        microsoft_client_secret: null,
        microsoft_tenant: null,
        generic_client_id: null,
        generic_client_secret: null,
        generic_authorization_endpoint: null,
        generic_token_endpoint: null,
        generic_userinfo_endpoint: null,
        proxy_base_url: null,
        user_email: null,
        sso_provider: null,
        role_mappings: null,
      };

      await updateSSOSettings(accessToken, clearSettings);

      rhfForm.reset(defaultFormValues);

      setIsClearConfirmModalVisible(false);

      handleAddSSOOk();

      NotificationsManager.success("SSO settings cleared successfully");
    } catch (error) {
      console.error("Failed to clear SSO settings:", error);
      NotificationsManager.fromBackend("Failed to clear SSO settings");
    }
  };

  return (
    <>
      <Dialog
        open={isAddSSOModalVisible}
        onOpenChange={(o) => {
          if (!o) handleAddSSOCancel();
        }}
      >
        <DialogContent className="max-w-[800px]">
          <DialogHeader>
            <DialogTitle>{ssoConfigured ? "Edit SSO Settings" : "Add SSO"}</DialogTitle>
          </DialogHeader>
          <FormProvider {...rhfForm}>
            <form onSubmit={onSubmit} className="space-y-4">
              <Controller
                control={rhfForm.control}
                name="sso_provider"
                rules={{ required: "Please select an SSO provider" }}
                render={({ field, fieldState }) => (
                  <div className="grid grid-cols-[1fr_2fr] gap-4 items-start">
                    <Label htmlFor="sso-provider" className="mt-2 text-left">
                      SSO Provider <span className="text-destructive">*</span>
                    </Label>
                    <div className="space-y-1">
                      <Select value={(field.value as string) || ""} onValueChange={field.onChange}>
                        <SelectTrigger id="sso-provider" aria-label="SSO Provider">
                          <SelectValue placeholder="Select an SSO provider" />
                        </SelectTrigger>
                        <SelectContent>
                          {Object.entries(ssoProviderLogoMap).map(([value, logo]) => {
                            const label =
                              value.toLowerCase() === "okta"
                                ? "Okta / Auth0"
                                : value.charAt(0).toUpperCase() + value.slice(1);
                            return (
                              <SelectItem key={value} value={value}>
                                <div className="flex items-center gap-3">
                                  {logo && (
                                    <img
                                      src={logo}
                                      alt={value}
                                      style={{ height: 24, width: 24, objectFit: "contain" }}
                                    />
                                  )}
                                  <span>{label} SSO</span>
                                </div>
                              </SelectItem>
                            );
                          })}
                        </SelectContent>
                      </Select>
                      {fieldState.error && (
                        <p className="text-sm text-destructive">{fieldState.error.message}</p>
                      )}
                    </div>
                  </div>
                )}
              />

              {provider ? <ProviderFields provider={provider} /> : null}

              <Controller
                control={rhfForm.control}
                name="user_email"
                rules={{ required: "Please enter the email of the proxy admin" }}
                render={({ field, fieldState }) => (
                  <div className="grid grid-cols-[1fr_2fr] gap-4 items-start">
                    <Label htmlFor="sso-user-email" className="mt-2 text-left">
                      Proxy Admin Email <span className="text-destructive">*</span>
                    </Label>
                    <div className="space-y-1">
                      <Input
                        id="sso-user-email"
                        value={(field.value as string) ?? ""}
                        onChange={field.onChange}
                      />
                      {fieldState.error && (
                        <p className="text-sm text-destructive">{fieldState.error.message}</p>
                      )}
                    </div>
                  </div>
                )}
              />

              <Controller
                control={rhfForm.control}
                name="proxy_base_url"
                rules={{
                  required: "Please enter the proxy base url",
                  validate: (rawValue) => {
                    const value = (rawValue ?? "").trim();
                    if (!URL_PATTERN.test(value)) {
                      return "URL must start with http:// or https://";
                    }
                    if (value.endsWith("/")) {
                      return "URL must not end with a trailing slash";
                    }
                    return true;
                  },
                }}
                render={({ field, fieldState }) => (
                  <div className="grid grid-cols-[1fr_2fr] gap-4 items-start">
                    <Label htmlFor="sso-proxy-base-url" className="mt-2 text-left">
                      Proxy Base URL <span className="text-destructive">*</span>
                    </Label>
                    <div className="space-y-1">
                      <Input
                        id="sso-proxy-base-url"
                        placeholder="https://example.com"
                        value={(field.value as string) ?? ""}
                        onChange={(e) => field.onChange(e.target.value)}
                        onBlur={(e) => {
                          field.onChange(e.target.value.trim());
                          field.onBlur();
                        }}
                      />
                      {fieldState.error && (
                        <p className="text-sm text-destructive">{fieldState.error.message}</p>
                      )}
                    </div>
                  </div>
                )}
              />

              {(provider === "okta" || provider === "generic") && (
                <Controller
                  control={rhfForm.control}
                  name="use_role_mappings"
                  render={({ field }) => (
                    <div className="grid grid-cols-[1fr_2fr] gap-4 items-center">
                      <Label htmlFor="sso-use-role-mappings" className="text-left">
                        Use Role Mappings
                      </Label>
                      <Checkbox
                        id="sso-use-role-mappings"
                        checked={!!field.value}
                        onCheckedChange={(c) => field.onChange(c === true)}
                        aria-label="Use Role Mappings"
                      />
                    </div>
                  )}
                />
              )}

              {useRoleMappings ? <RoleMappingFields /> : null}

              <div className="flex justify-end items-center gap-2 pt-3">
                {ssoConfigured && (
                  <Button
                    type="button"
                    variant="secondary"
                    onClick={() => setIsClearConfirmModalVisible(true)}
                  >
                    Clear
                  </Button>
                )}
                <Button type="submit">Save</Button>
              </div>
            </form>
          </FormProvider>
        </DialogContent>
      </Dialog>

      <Dialog
        open={isClearConfirmModalVisible}
        onOpenChange={(o) => {
          if (!o) setIsClearConfirmModalVisible(false);
        }}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Confirm Clear SSO Settings</DialogTitle>
          </DialogHeader>
          <div className="space-y-2">
            <p>Are you sure you want to clear all SSO settings? This action cannot be undone.</p>
            <p>Users will no longer be able to login using SSO after this change.</p>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setIsClearConfirmModalVisible(false)}>
              Cancel
            </Button>
            <Button variant="destructive" onClick={handleClearSSO}>
              Yes, Clear
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog
        open={isInstructionsModalVisible}
        onOpenChange={(o) => {
          if (!o) handleInstructionsCancel();
        }}
      >
        <DialogContent className="max-w-[800px]">
          <DialogHeader>
            <DialogTitle>SSO Setup Instructions</DialogTitle>
          </DialogHeader>
          <div className="space-y-2">
            <p>Follow these steps to complete the SSO setup:</p>
            <p className="mt-2">1. DO NOT Exit this TAB</p>
            <p className="mt-2">2. Open a new tab, visit your proxy base url</p>
            <p className="mt-2">3. Confirm your SSO is configured correctly and you can login on the new Tab</p>
            <p className="mt-2">4. If Step 3 is successful, you can close this tab</p>
          </div>
          <DialogFooter>
            <Button onClick={handleInstructionsOk}>Done</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
};

export { ssoProviderConfigs };
export default SSOModals;
