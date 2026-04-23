import React from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import NotificationManager from "./molecules/notifications_manager";
import { serviceHealthCheck, setCallbacksCall } from "./networking";
import { EmailEventSettings } from "./email_events";

interface EmailSettingsProps {
  accessToken: string | null;
  premiumUser: boolean;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  alerts: any[];
}

const EmailSettings: React.FC<EmailSettingsProps> = ({
  accessToken,
  premiumUser,
  alerts,
}) => {
  const handleSaveEmailSettings = async () => {
    if (!accessToken) return;

    const updatedVariables: Record<string, string> = {};

    alerts
      .filter((alert) => alert.name === "email")
      .forEach((alert) => {
        Object.entries(alert.variables ?? {}).forEach(([key]) => {
          const inputElement = document.querySelector(
            `input[name="${key}"]`,
          ) as HTMLInputElement;
          if (inputElement && inputElement.value) {
            updatedVariables[key] = inputElement.value;
          }
        });
      });

    const payload = {
      general_settings: { alerting: ["email"] },
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
      <Card className="p-6">
        <h4 className="text-lg font-semibold m-0">Email Server Settings</h4>
        <p className="text-sm">
          <a
            href="https://docs.litellm.ai/docs/proxy/email"
            target="_blank"
            rel="noopener noreferrer"
            className="text-primary hover:underline"
          >
            LiteLLM Docs: email alerts
          </a>
        </p>

        <div className="flex w-full mt-4">
          {alerts
            .filter((alert) => alert.name === "email")
            .map((alert, index) => (
              <div key={index} className="w-full">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-x-4">
                  {Object.entries(alert.variables ?? {}).map(
                    ([key, value]) => (
                      <div key={key} className="mx-2 my-2 space-y-1">
                        {premiumUser !== true &&
                        (key === "EMAIL_LOGO_URL" ||
                          key === "EMAIL_SUPPORT_CONTACT") ? (
                          <div>
                            <a
                              href="https://forms.gle/W3U4PZpJGFHWtHyA9"
                              target="_blank"
                              rel="noopener noreferrer"
                            >
                              <Label className="mt-2">✨ {key}</Label>
                            </a>
                            <Input
                              name={key}
                              defaultValue={value as string}
                              type="password"
                              disabled
                              className="w-full max-w-[400px]"
                            />
                          </div>
                        ) : (
                          <div>
                            <Label className="mt-2">{key}</Label>
                            <Input
                              name={key}
                              defaultValue={value as string}
                              type="password"
                              className="w-full max-w-[400px]"
                            />
                          </div>
                        )}

                        <p className="text-xs italic text-muted-foreground">
                          {key === "SMTP_HOST" && (
                            <span>
                              Enter the SMTP host address, e.g.{" "}
                              <code>smtp.resend.com</code>
                              <span className="text-destructive">
                                {" "}
                                Required *
                              </span>
                            </span>
                          )}
                          {key === "SMTP_PORT" && (
                            <span>
                              Enter the SMTP port number, e.g. <code>587</code>
                              <span className="text-destructive">
                                {" "}
                                Required *
                              </span>
                            </span>
                          )}
                          {key === "SMTP_USERNAME" && (
                            <span>
                              Enter the SMTP username, e.g. <code>username</code>
                              <span className="text-destructive">
                                {" "}
                                Required *
                              </span>
                            </span>
                          )}
                          {key === "SMTP_PASSWORD" && (
                            <span className="text-destructive">
                              {" "}
                              Required *
                            </span>
                          )}
                          {key === "SMTP_SENDER_EMAIL" && (
                            <span>
                              Enter the sender email address, e.g.{" "}
                              <code>sender@berri.ai</code>
                              <span className="text-destructive">
                                {" "}
                                Required *
                              </span>
                            </span>
                          )}
                          {key === "TEST_EMAIL_ADDRESS" && (
                            <span>
                              Email Address to send <code>Test Email Alert</code>{" "}
                              to. example: <code>info@berri.ai</code>
                              <span className="text-destructive">
                                {" "}
                                Required *
                              </span>
                            </span>
                          )}
                          {key === "EMAIL_LOGO_URL" && (
                            <span>
                              (Optional) Customize the Logo that appears in the
                              email, pass a url to your logo
                            </span>
                          )}
                          {key === "EMAIL_SUPPORT_CONTACT" && (
                            <span>
                              (Optional) Customize the support email address that
                              appears in the email. Default is support@berri.ai
                            </span>
                          )}
                        </p>
                      </div>
                    ),
                  )}
                </div>
              </div>
            ))}
        </div>

        <div className="flex gap-2 mt-4">
          <Button onClick={() => handleSaveEmailSettings()}>Save Changes</Button>
          <Button
            variant="outline"
            onClick={async () => {
              if (!accessToken) return;
              try {
                await serviceHealthCheck(accessToken, "email");
                NotificationManager.success(
                  "Email test triggered. Check your configured email inbox/logs.",
                );
              } catch (error) {
                NotificationManager.fromBackend(error);
              }
            }}
          >
            Test Email Alerts
          </Button>
        </div>
      </Card>
    </>
  );
};

export default EmailSettings;
