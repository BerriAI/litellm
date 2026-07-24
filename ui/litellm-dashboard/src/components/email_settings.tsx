import React from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import NotificationManager from "./molecules/notifications_manager";
import { serviceHealthCheck, setCallbacksCall } from "./networking";
import { EmailEventSettings } from "./email_events";

interface EmailSettingsProps {
  accessToken: string | null;
  premiumUser: boolean;
  alerts: any[];
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

const EmailSettings: React.FC<EmailSettingsProps> = ({ accessToken, premiumUser, alerts }) => {
  const handleSaveEmailSettings = async () => {
    if (!accessToken) {
      return;
    }

    let updatedVariables: Record<string, string> = {};

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
      NotificationManager.success("Email settings updated successfully");
    } catch (error) {
      NotificationManager.fromBackend(error);
    }
  };

  return (
    <>
      <div className="mt-6 mb-6">
        <EmailEventSettings accessToken={accessToken} />
      </div>
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
                  return (
                    <div key={key} className="space-y-1">
                      {isLocked ? (
                        <a
                          href="https://forms.gle/W3U4PZpJGFHWtHyA9"
                          target="_blank"
                          rel="noreferrer"
                          className="text-sm text-primary underline underline-offset-4"
                        >
                          ✨ {key}
                        </a>
                      ) : (
                        <p className="text-sm">{key}</p>
                      )}
                      <Input
                        name={key}
                        defaultValue={value as string}
                        type="password"
                        disabled={isLocked}
                        className="max-w-100"
                      />
                      <div className="text-xs text-muted-foreground italic">{FIELD_HELP[key]}</div>
                    </div>
                  );
                })}
              </div>
            ))}

          <div className="mt-6 flex gap-2">
            <Button onClick={() => handleSaveEmailSettings()}>Save Changes</Button>
            <Button
              variant="secondary"
              onClick={async () => {
                if (!accessToken) return;
                try {
                  await serviceHealthCheck(accessToken, "email");
                  NotificationManager.success("Email test triggered. Check your configured email inbox/logs.");
                } catch (error) {
                  NotificationManager.fromBackend(error);
                }
              }}
            >
              Test Email Alerts
            </Button>
          </div>
        </CardContent>
      </Card>
    </>
  );
};

export default EmailSettings;
