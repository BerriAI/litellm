import React, { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { InputGroup, InputGroupAddon, InputGroupButton, InputGroupInput } from "@/components/ui/input-group";
import { SimpleTooltip } from "@/components/ui/tooltip";
import { Eye, EyeOff } from "lucide-react";
import { toast } from "@/lib/toast";
import { isProxyAdminRole } from "@/utils/roles";
import {
  getSpendReportEmailStatus,
  serviceHealthCheck,
  setCallbacksCall,
  type SpendReportEmailStatusResponse,
} from "./networking";
import { EmailEventSettings } from "./email_events";
import SpendReportEmailSettings from "./SpendReportEmailSettings";

interface EmailSettingsProps {
  accessToken: string | null;
  premiumUser: boolean;
  alerts: any[];
  userRole?: string | null;
}

const REQUIRED_MARKER = <span className="text-destructive"> Required * </span>;

const FIELD_HELP: Record<string, React.ReactNode> = {
  SMTP_HOST: <>Enter the SMTP host address, e.g. `smtp.resend.com`{REQUIRED_MARKER}</>,
  SMTP_PORT: <>Enter the SMTP port number, e.g. `587`{REQUIRED_MARKER}</>,
  SMTP_USERNAME: <>Enter the SMTP username, e.g. `username`{REQUIRED_MARKER}</>,
  SMTP_PASSWORD: REQUIRED_MARKER,
  SMTP_SENDER_EMAIL: <>Enter the sender email address, e.g. `sender@berri.ai`{REQUIRED_MARKER}</>,
  TEST_EMAIL_ADDRESS: <>Email Address to send `Test Email Alert` to. example: `info@berri.ai`{REQUIRED_MARKER}</>,
  EMAIL_LOGO_URL: <>(Optional) Customize the Logo that appears in the email, pass a url to your logo</>,
  EMAIL_SUPPORT_CONTACT: (
    <>(Optional) Customize the support email address that appears in the email. Default is support@berri.ai</>
  ),
};

const PREMIUM_ONLY_FIELDS = ["EMAIL_LOGO_URL", "EMAIL_SUPPORT_CONTACT"];

const SENSITIVE_FIELD_PATTERN = /(PASSWORD|SECRET|KEY|TOKEN)/i;

const EmailSettings: React.FC<EmailSettingsProps> = ({ accessToken, premiumUser, alerts, userRole }) => {
  const [visibleFields, setVisibleFields] = useState<Record<string, boolean>>({});
  const [emailStatus, setEmailStatus] = useState<SpendReportEmailStatusResponse | null>(null);
  const isProxyAdmin = isProxyAdminRole(userRole || "");

  useEffect(() => {
    if (!accessToken || !isProxyAdmin) return;
    let active = true;
    getSpendReportEmailStatus(accessToken)
      .then((status) => {
        if (active) setEmailStatus(status);
      })
      .catch(() => {
        if (active) setEmailStatus(null);
      });
    return () => {
      active = false;
    };
  }, [accessToken, isProxyAdmin]);

  const toggleFieldVisibility = (key: string) => {
    setVisibleFields((prev) => ({
      ...prev,
      [key]: !prev[key],
    }));
  };

  const handleSaveEmailSettings = async () => {
    if (!accessToken) {
      return;
    }

    const updatedVariables: Record<string, string> = {};

    alerts
      .filter((alert) => alert.name === "email")
      .forEach((alert) => {
        Object.entries(alert.variables ?? {}).forEach(([key, value]) => {
          const inputElement = document.querySelector(`input[name="${key}"]`) as HTMLInputElement;
          if (!inputElement || !inputElement.value) {
            return;
          }
          // Only send fields the admin actually edited. Values rendered from the
          // server are masked (SMTP_PASSWORD) or sourced from the process
          // environment, so re-submitting an untouched field would persist a mask
          // or copy env-managed config into the database.
          if (inputElement.value === (value == null ? "" : String(value))) {
            return;
          }
          updatedVariables[key] = inputElement.value;
        });
      });

    //filter out null / undefined values for updatedVariables

    const payload = {
      general_settings: {
        alerting: ["email"],
      },
      environment_variables: updatedVariables,
    };
    try {
      await setCallbacksCall(accessToken, payload);
      toast.success("Email settings updated successfully");
      if (isProxyAdmin) {
        getSpendReportEmailStatus(accessToken)
          .then(setEmailStatus)
          .catch(() => setEmailStatus(null));
      }
    } catch (error) {
      toast.fromError(error);
    }
  };

  return (
    <>
      <div className="mb-6">
        <EmailEventSettings accessToken={accessToken} />
      </div>

      {isProxyAdmin && (
        <div className="mt-6 mb-6">
          <SpendReportEmailSettings accessToken={accessToken} userRole={userRole} />
        </div>
      )}

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Email Server Settings</CardTitle>
          <p className="text-sm">
            <a
              href="https://docs.litellm.ai/docs/proxy/email"
              target="_blank"
              rel="noreferrer"
              className="text-primary underline underline-offset-4"
            >
              LiteLLM Docs: email alerts
            </a>
          </p>
        </CardHeader>

        <CardContent>
          {alerts
            .filter((alert) => alert.name === "email")
            .map((alert, index) => (
              <div key={index} className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                {Object.entries(alert.variables ?? {}).map(([key, value]) => {
                  const isLocked = !premiumUser && PREMIUM_ONLY_FIELDS.includes(key);
                  const isSensitive = SENSITIVE_FIELD_PATTERN.test(key);
                  const isVisible = visibleFields[key] || false;
                  return (
                    <div key={key} className="space-y-1">
                      {isLocked ? (
                        <a
                          href="https://forms.gle/W3U4PZpJGFHWtHyA9"
                          target="_blank"
                          rel="noreferrer"
                          className="inline-flex items-center gap-1.5 text-sm text-primary underline underline-offset-4"
                        >
                          ✨ {key}
                          <span className="rounded bg-primary/10 px-1.5 py-0.5 text-[10px] font-medium text-primary no-underline">
                            Enterprise Feature
                          </span>
                        </a>
                      ) : (
                        <p className="text-sm">{key}</p>
                      )}
                      <InputGroup className="max-w-100">
                        <InputGroupInput
                          name={key}
                          defaultValue={value as string}
                          type={isSensitive && !isVisible ? "password" : "text"}
                          disabled={isLocked}
                        />
                        {isSensitive && (
                          <InputGroupAddon align="inline-end">
                            <InputGroupButton
                              size="icon-xs"
                              onClick={() => toggleFieldVisibility(key)}
                              aria-label={isVisible ? "Hide credential" : "Show credential"}
                            >
                              {isVisible ? <EyeOff /> : <Eye />}
                            </InputGroupButton>
                          </InputGroupAddon>
                        )}
                      </InputGroup>
                      <div className="text-xs text-muted-foreground italic">{FIELD_HELP[key]}</div>
                    </div>
                  );
                })}
              </div>
            ))}

          <div className="mt-6 flex gap-2">
            <Button onClick={() => handleSaveEmailSettings()}>Save Changes</Button>
            <SimpleTooltip
              content={
                emailStatus && !emailStatus.configured
                  ? "Email delivery is not configured. Configure SMTP server settings below or set SENDGRID_API_KEY environment variable."
                  : undefined
              }
            >
              <Button
                variant="secondary"
                disabled={emailStatus !== null && !emailStatus.configured}
                onClick={async () => {
                  if (!accessToken) return;
                  try {
                    await serviceHealthCheck(accessToken, "email");
                    toast.success("Email test triggered. Check your configured email inbox/logs.");
                  } catch (error) {
                    toast.fromError(error);
                  }
                }}
              >
                Test Email Alerts
              </Button>
            </SimpleTooltip>
          </div>
        </CardContent>
      </Card>
    </>
  );
};

export default EmailSettings;
