import React, { useState, useEffect } from "react";
import {
  Alert,
  AlertDescription,
  AlertTitle,
} from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Separator } from "@/components/ui/separator";
import { keyCreateCall } from "./networking";
import { CopyToClipboard } from "react-copy-to-clipboard";
import {
  AlertCircle,
  Copy,
  Key,
  Link as LinkIcon,
  PlusCircle,
} from "lucide-react";
import { parseErrorMessage } from "./shared/errorUtils";
import NotificationsManager from "./molecules/notifications_manager";
import { Controller, FormProvider, useForm } from "react-hook-form";

interface SCIMConfigProps {
  accessToken: string | null;
  userID: string | null;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  proxySettings: any;
}

interface SCIMFormValues {
  key_alias: string;
}

const SCIMConfig: React.FC<SCIMConfigProps> = ({
  accessToken,
  userID,
  proxySettings,
}) => {
  const form = useForm<SCIMFormValues>({
    defaultValues: { key_alias: "" },
    mode: "onSubmit",
  });
  const [isCreatingToken, setIsCreatingToken] = useState(false);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const [tokenData, setTokenData] = useState<any>(null);
  const [baseUrl, setBaseUrl] = useState("<your_proxy_base_url>");

  useEffect(() => {
    let url = "<your_proxy_base_url>";
    if (
      proxySettings &&
      proxySettings.PROXY_BASE_URL &&
      proxySettings.PROXY_BASE_URL !== undefined
    ) {
      url = proxySettings.PROXY_BASE_URL;
    } else if (typeof window !== "undefined") {
      url = window.location.origin;
    }
    setBaseUrl(url);
  }, [proxySettings]);

  const scimBaseUrl = `${baseUrl}/scim/v2`;

  const handleCreateSCIMToken = form.handleSubmit(async (values) => {
    if (!accessToken || !userID) {
      NotificationsManager.fromBackend(
        "You need to be logged in to create a SCIM token",
      );
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
      NotificationsManager.success("SCIM token created successfully");
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
    } catch (error: any) {
      console.error("Error creating SCIM token:", error);
      NotificationsManager.fromBackend(
        "Failed to create SCIM token: " + parseErrorMessage(error),
      );
    } finally {
      setIsCreatingToken(false);
    }
  });

  return (
    <Card className="p-6">
      <div className="flex items-center mb-4">
        <h3 className="text-lg font-semibold">SCIM Configuration</h3>
      </div>
      <p className="text-sm text-muted-foreground">
        System for Cross-domain Identity Management (SCIM) allows you to
        automatically provision and manage users and groups in LiteLLM.
      </p>

      <Separator className="my-6" />

      <div className="space-y-8">
        {/* Step 1: SCIM URL */}
        <div>
          <div className="flex items-center mb-2">
            <div className="flex items-center justify-center w-6 h-6 rounded-full bg-primary/15 text-primary mr-2 text-sm font-medium">
              1
            </div>
            <h4 className="text-lg font-semibold flex items-center">
              <LinkIcon className="h-5 w-5 mr-2" />
              SCIM Tenant URL
            </h4>
          </div>
          <p className="text-sm text-muted-foreground mb-3">
            Use this URL in your identity provider SCIM integration settings.
          </p>
          <div className="flex items-center gap-2">
            <Input value={scimBaseUrl} disabled className="flex-grow" />
            <CopyToClipboard
              text={scimBaseUrl}
              onCopy={() =>
                NotificationsManager.success("URL copied to clipboard")
              }
            >
              <Button>
                <Copy className="h-4 w-4" />
                Copy
              </Button>
            </CopyToClipboard>
          </div>
        </div>

        {/* Step 2: SCIM Token */}
        <div>
          <div className="flex items-center mb-2">
            <div className="flex items-center justify-center w-6 h-6 rounded-full bg-primary/15 text-primary mr-2 text-sm font-medium">
              2
            </div>
            <h4 className="text-lg font-semibold flex items-center">
              <Key className="h-5 w-5 mr-2" />
              Authentication Token
            </h4>
          </div>

          <Alert className="mb-4">
            <AlertTitle>Using SCIM</AlertTitle>
            <AlertDescription>
              You need a SCIM token to authenticate with the SCIM API. Create
              one below and use it in your SCIM provider configuration.
            </AlertDescription>
          </Alert>

          {!tokenData ? (
            <div className="bg-muted p-4 rounded-lg">
              <FormProvider {...form}>
                <form onSubmit={handleCreateSCIMToken} className="space-y-4">
                  <div className="space-y-2">
                    <Label htmlFor="key_alias">
                      Token Name <span className="text-destructive">*</span>
                    </Label>
                    <Controller
                      control={form.control}
                      name="key_alias"
                      rules={{
                        required: "Please enter a name for your token",
                      }}
                      render={({ field }) => (
                        <Input
                          id="key_alias"
                          placeholder="SCIM Access Token"
                          {...field}
                        />
                      )}
                    />
                    {form.formState.errors.key_alias && (
                      <p className="text-sm text-destructive">
                        {form.formState.errors.key_alias.message as string}
                      </p>
                    )}
                  </div>
                  <Button type="submit" disabled={isCreatingToken}>
                    <Key className="h-4 w-4" />
                    {isCreatingToken ? "Creating…" : "Create SCIM Token"}
                  </Button>
                </form>
              </FormProvider>
            </div>
          ) : (
            <Card className="p-4 border-amber-300 bg-amber-50 dark:bg-amber-950/30 dark:border-amber-800">
              <div className="flex items-center mb-2 text-amber-800 dark:text-amber-200">
                <AlertCircle className="h-5 w-5 mr-2" />
                <h4 className="text-lg font-semibold">Your SCIM Token</h4>
              </div>
              <p className="text-sm font-medium text-amber-800 dark:text-amber-200 mb-4">
                Make sure to copy this token now. You will not be able to see
                it again.
              </p>
              <div className="flex items-center gap-2">
                <Input
                  value={tokenData.key}
                  className="flex-grow bg-background"
                  type="password"
                  disabled
                />
                <CopyToClipboard
                  text={tokenData.key}
                  onCopy={() =>
                    NotificationsManager.success("Token copied to clipboard")
                  }
                >
                  <Button>
                    <Copy className="h-4 w-4" />
                    Copy
                  </Button>
                </CopyToClipboard>
              </div>
              <Button
                variant="secondary"
                className="mt-4"
                onClick={() => setTokenData(null)}
              >
                <PlusCircle className="h-4 w-4" />
                Create Another Token
              </Button>
            </Card>
          )}
        </div>
      </div>
    </Card>
  );
};

export default SCIMConfig;
