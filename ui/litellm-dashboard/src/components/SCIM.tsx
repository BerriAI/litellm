import React, { useState, useEffect } from "react";
import { z } from "zod/v4";
import { getScimSettings, keyCreateCall, updateScimSettings } from "./networking";
import { CopyToClipboard } from "react-copy-to-clipboard";
import { Building2, CircleAlert, CirclePlus, Copy, Info, KeyRound, Link, Trash2 } from "lucide-react";
import { parseErrorMessage } from "./shared/errorUtils";
import { toast } from "@/lib/toast";
import { Alert, AlertDescription, AlertTitle } from "@/components/shared/Alert";
import { FieldGroup } from "@/components/ui/field";
import { FormField } from "@/components/shared/form/FormField";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Separator } from "@/components/ui/separator";
import { UiLoadingSpinner } from "@/components/ui/ui-loading-spinner";
import { useZodForm } from "@/lib/forms/useZodForm";

interface SCIMConfigProps {
  accessToken: string | null;
  userID: string | null;
  proxySettings: any;
}

const scimTokenSchema = z.object({
  key_alias: z.string().min(1, "Please enter a name for your token"),
});

type SCIMTokenFormValues = z.infer<typeof scimTokenSchema>;

interface SCIMOrganizationMapping {
  group_display_name_pattern: string;
  organization_id: string;
}

const SCIMConfig: React.FC<SCIMConfigProps> = ({ accessToken, userID, proxySettings }) => {
  const form = useZodForm(scimTokenSchema, { defaultValues: { key_alias: "" } });
  const [isCreatingToken, setIsCreatingToken] = useState(false);
  const [tokenData, setTokenData] = useState<any>(null);
  const [baseUrl, setBaseUrl] = useState("<your_proxy_base_url>");
  const [orgMappings, setOrgMappings] = useState<SCIMOrganizationMapping[]>([]);
  const [isSavingMappings, setIsSavingMappings] = useState(false);

  useEffect(() => {
    if (!accessToken) return;
    getScimSettings(accessToken)
      .then((data) => {
        const mappings = data?.values?.organization_mappings;
        if (Array.isArray(mappings)) {
          setOrgMappings(mappings);
        }
      })
      .catch((error) => console.error("Failed to load SCIM settings:", error));
  }, [accessToken]);

  useEffect(() => {
    let url = "<your_proxy_base_url>";

    if (proxySettings && proxySettings.PROXY_BASE_URL && proxySettings.PROXY_BASE_URL !== undefined) {
      url = proxySettings.PROXY_BASE_URL;
    } else if (typeof window !== "undefined") {
      // Use the current origin as the base URL if no proxy URL is set
      url = window.location.origin;
    }

    setBaseUrl(url);
  }, [proxySettings]);

  const scimBaseUrl = `${baseUrl}/scim/v2`;

  const handleSaveOrgMappings = async () => {
    if (!accessToken) {
      toast.fromError("You need to be logged in to update SCIM settings");
      return;
    }
    const incomplete = orgMappings.some((m) => !m.group_display_name_pattern.trim() || !m.organization_id.trim());
    if (incomplete) {
      toast.fromError("Each mapping needs both a group name pattern and an organization ID");
      return;
    }
    try {
      setIsSavingMappings(true);
      await updateScimSettings(accessToken, { organization_mappings: orgMappings });
      toast.success("SCIM organization mappings saved");
    } catch (error: any) {
      console.error("Error saving SCIM settings:", error);
      toast.fromError("Failed to save SCIM settings: " + parseErrorMessage(error));
    } finally {
      setIsSavingMappings(false);
    }
  };

  const handleCreateSCIMToken = async (values: SCIMTokenFormValues) => {
    if (!accessToken || !userID) {
      toast.fromError("You need to be logged in to create a SCIM token");
      return;
    }

    try {
      setIsCreatingToken(true);

      const formData = {
        key_alias: values.key_alias || "SCIM Access Token",
        team_id: null,
        models: [],
        allowed_routes: ["/scim/*"],
      };

      const response = await keyCreateCall(accessToken, userID, formData);
      setTokenData(response);
      toast.success("SCIM token created successfully");
    } catch (error: any) {
      console.error("Error creating SCIM token:", error);
      toast.fromError("Failed to create SCIM token: " + parseErrorMessage(error));
    } finally {
      setIsCreatingToken(false);
    }
  };

  return (
    <div className="grid grid-cols-1">
      <Card>
        <CardContent>
          <div className="flex items-center mb-4">
            <CardTitle>SCIM Configuration</CardTitle>
          </div>
          <p className="text-muted-foreground">
            System for Cross-domain Identity Management (SCIM) allows you to automatically provision and manage users
            and groups in LiteLLM.
          </p>

          <Separator className="my-6" />

          <div className="space-y-8">
            {/* Step 1: SCIM URL */}
            <div>
              <div className="flex items-center mb-2">
                <div className="flex items-center justify-center w-6 h-6 rounded-full bg-info/15 text-info mr-2">1</div>
                <h3 className="text-lg font-medium flex items-center">
                  <Link className="h-5 w-5 mr-2" />
                  SCIM Tenant URL
                </h3>
              </div>
              <p className="text-muted-foreground mb-3">
                Use this URL in your identity provider SCIM integration settings.
              </p>
              <div className="flex items-center">
                <Input value={scimBaseUrl} disabled={true} readOnly className="grow" />
                <CopyToClipboard text={scimBaseUrl} onCopy={() => toast.success("URL copied to clipboard")}>
                  <Button type="button" className="ml-2 flex items-center">
                    <Copy />
                    Copy
                  </Button>
                </CopyToClipboard>
              </div>
            </div>

            {/* Step 2: SCIM Token */}
            <div>
              <div className="flex items-center mb-2">
                <div className="flex items-center justify-center w-6 h-6 rounded-full bg-info/15 text-info mr-2">2</div>
                <h3 className="text-lg font-medium flex items-center">
                  <KeyRound className="h-5 w-5 mr-2" />
                  Authentication Token
                </h3>
              </div>

              <Alert variant="info" className="mb-4">
                <Info />
                <AlertTitle>Using SCIM</AlertTitle>
                <AlertDescription>
                  You need a SCIM token to authenticate with the SCIM API. Create one below and use it in your SCIM
                  provider configuration.
                </AlertDescription>
              </Alert>

              {!tokenData ? (
                <div className="bg-muted p-4 rounded-lg">
                  <form onSubmit={form.handleSubmit(handleCreateSCIMToken)}>
                    <FieldGroup>
                      <FormField control={form.control} name="key_alias" label="Token Name">
                        {({ ref, ...field }) => <Input {...field} ref={ref} placeholder="SCIM Access Token" />}
                      </FormField>
                      <div>
                        <Button
                          type="submit"
                          disabled={isCreatingToken}
                          aria-busy={isCreatingToken}
                          className="flex items-center"
                        >
                          {isCreatingToken ? <UiLoadingSpinner className="size-4" /> : <KeyRound />}
                          Create SCIM Token
                        </Button>
                      </div>
                    </FieldGroup>
                  </form>
                </div>
              ) : (
                <Card className="block p-6 border border-warning/30 bg-warning/10">
                  <div className="flex items-center mb-2 text-warning">
                    <CircleAlert className="h-5 w-5 mr-2" />
                    <h4 className="text-lg font-medium text-warning">Your SCIM Token</h4>
                  </div>
                  <p className="text-warning mb-4 font-medium">
                    Make sure to copy this token now. You will not be able to see it again.
                  </p>
                  <div className="flex items-center">
                    <Input value={tokenData.key} className="grow mr-2" type="password" disabled={true} readOnly />
                    <CopyToClipboard text={tokenData.key} onCopy={() => toast.success("Token copied to clipboard")}>
                      <Button type="button" className="flex items-center">
                        <Copy />
                        Copy
                      </Button>
                    </CopyToClipboard>
                  </div>
                  <Button
                    type="button"
                    variant="secondary"
                    className="mt-4 flex items-center"
                    onClick={() => setTokenData(null)}
                  >
                    <CirclePlus />
                    Create Another Token
                  </Button>
                </Card>
              )}
            </div>

            {/* Step 3: Organization Mappings */}
            <div>
              <div className="flex items-center mb-2">
                <div className="flex items-center justify-center w-6 h-6 rounded-full bg-info/15 text-info mr-2">3</div>
                <h3 className="text-lg font-medium flex items-center">
                  <Building2 className="h-5 w-5 mr-2" />
                  Organization Mappings
                </h3>
              </div>
              <p className="text-muted-foreground mb-3">
                Automatically assign SCIM-provisioned teams to organizations based on their group display name. The
                pattern is a regex fully matched against the SCIM group displayName; a plain group name works as an
                exact match. The first matching entry wins.
              </p>
              <div className="space-y-2">
                {orgMappings.map((mapping, index) => (
                  <div key={index} className="flex items-center gap-2">
                    <Input
                      value={mapping.group_display_name_pattern}
                      placeholder="Group display name pattern (e.g. Engineering-.*)"
                      aria-label="Group display name pattern"
                      onChange={(e) =>
                        setOrgMappings((prev) =>
                          prev.map((m, i) => (i === index ? { ...m, group_display_name_pattern: e.target.value } : m)),
                        )
                      }
                    />
                    <Input
                      value={mapping.organization_id}
                      placeholder="Organization ID"
                      aria-label="Organization ID"
                      onChange={(e) =>
                        setOrgMappings((prev) =>
                          prev.map((m, i) => (i === index ? { ...m, organization_id: e.target.value } : m)),
                        )
                      }
                    />
                    <Button
                      type="button"
                      variant="secondary"
                      aria-label="Remove mapping"
                      onClick={() => setOrgMappings((prev) => prev.filter((_, i) => i !== index))}
                    >
                      <Trash2 />
                    </Button>
                  </div>
                ))}
                <div className="flex items-center gap-2">
                  <Button
                    type="button"
                    variant="secondary"
                    className="flex items-center"
                    onClick={() =>
                      setOrgMappings((prev) => [...prev, { group_display_name_pattern: "", organization_id: "" }])
                    }
                  >
                    <CirclePlus />
                    Add Mapping
                  </Button>
                  <Button
                    type="button"
                    disabled={isSavingMappings}
                    aria-busy={isSavingMappings}
                    className="flex items-center"
                    onClick={handleSaveOrgMappings}
                  >
                    {isSavingMappings ? <UiLoadingSpinner className="size-4" /> : null}
                    Save Mappings
                  </Button>
                </div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default SCIMConfig;
