import React from "react";
import { Card, Text, Grid, Button, TextInput, TableCell } from "@tremor/react";
import { Typography } from "antd";
import NotificationManager from "./molecules/notifications_manager";
import { serviceHealthCheck, setCallbacksCall } from "./networking";
import { EmailEventSettings } from "./email_events";

const { Title } = Typography;

interface EmailSettingsProps {
  accessToken: string | null;
  premiumUser: boolean;
  alerts: any[];
}

const EmailSettings: React.FC<EmailSettingsProps> = ({ accessToken, premiumUser, alerts }) => {
  const handleSaveEmailSettings = async ({ silent = false }: { silent?: boolean } = {}) => {
    if (!accessToken) {
      return;
    }

    let updatedVariables: Record<string, string> = {};

    alerts
      .filter((alert) => alert.name === "email")
      .forEach((alert) => {
        Object.entries(alert.variables ?? {}).forEach(([key, value]) => {
          const inputElement = document.querySelector(`input[name="${key}"]`) as HTMLInputElement;
          if (inputElement && inputElement.value) {
            updatedVariables[key] = inputElement?.value;
          }
        });
      });

    console.log("updatedVariables", updatedVariables);
    //filter out null / undefined values for updatedVariables

    const payload = {
      general_settings: {
        alerting: ["email"],
      },
      environment_variables: updatedVariables,
    };
    try {
      await setCallbacksCall(accessToken, payload);
      if (!silent) {
        NotificationManager.success("Email settings updated successfully");
      }
    } catch (error) {
      if (!silent) {
        NotificationManager.fromBackend(error);
      }
      // In silent mode (called from test flow) swallow the error so that
      // the test can still proceed using env-var / YAML config values.
    }
  };

  return (
    <>
      <div className="mt-6 mb-6">
        <EmailEventSettings accessToken={accessToken} />
      </div>
      <Card>
        <Title level={4}>Email Server Settings</Title>
        <Text>
          <a href="https://docs.litellm.ai/docs/proxy/email" target="_blank" style={{ color: "blue" }}>
            {" "}
            LiteLLM Docs: email alerts
          </a>{" "}
          <br />
        </Text>

        <div className="flex w-full">
          {alerts
            .filter((alert) => alert.name === "email")
            .map((alert, index) => (
              <TableCell key={index}>
                <ul>
                  <Grid numItems={2}>
                    {Object.entries(alert.variables ?? {}).map(([key, value]) => (
                      <li key={key} className="mx-2 my-2">
                        {premiumUser != true && (key === "EMAIL_LOGO_URL" || key === "EMAIL_SUPPORT_CONTACT") ? (
                          <div>
                            <a href="https://forms.gle/W3U4PZpJGFHWtHyA9" target="_blank">
                              <Text className="mt-2"> ✨ {key}</Text>
                            </a>
                            <TextInput
                              name={key}
                              defaultValue={value as string}
                              type="password"
                              disabled={true}
                              style={{ width: "400px" }}
                            />
                          </div>
                        ) : (
                          <div>
                            <Text className="mt-2">{key}</Text>
                            <TextInput
                              name={key}
                              defaultValue={value as string}
                              type="password"
                              style={{ width: "400px" }}
                            />
                          </div>
                        )}

                        {/* Added descriptions for input fields */}
                        <p style={{ fontSize: "small", fontStyle: "italic" }}>
                          {key === "SMTP_HOST" && (
                            <div style={{ color: "gray" }}>
                              Enter the SMTP host address, e.g. `smtp.resend.com`
                              <span style={{ color: "red" }}> Required * </span>
                            </div>
                          )}

                          {key === "SMTP_PORT" && (
                            <div style={{ color: "gray" }}>
                              Enter the SMTP port number, e.g. `587`
                              <span style={{ color: "red" }}> Required * </span>
                            </div>
                          )}

                          {key === "SMTP_USERNAME" && (
                            <div style={{ color: "gray" }}>
                              Enter the SMTP username, e.g. `username`
                              <span style={{ color: "red" }}> Required * </span>
                            </div>
                          )}

                          {key === "SMTP_PASSWORD" && <span style={{ color: "red" }}> Required * </span>}

                          {key === "SMTP_SENDER_EMAIL" && (
                            <div style={{ color: "gray" }}>
                              Enter the sender email address, e.g. `sender@berri.ai`
                              <span style={{ color: "red" }}> Required * </span>
                            </div>
                          )}

                          {key === "TEST_EMAIL_ADDRESS" && (
                            <div style={{ color: "gray" }}>
                              Email Address to send `Test Email Alert` to. example: `info@berri.ai`
                              <span style={{ color: "red" }}> Required * </span>
                            </div>
                          )}
                          {key === "EMAIL_LOGO_URL" && (
                            <div style={{ color: "gray" }}>
                              (Optional) Customize the Logo that appears in the email, pass a url to your logo
                            </div>
                          )}
                          {key === "EMAIL_SUPPORT_CONTACT" && (
                            <div style={{ color: "gray" }}>
                              (Optional) Customize the support email address that appears in the email. Default is
                              support@berri.ai
                            </div>
                          )}
                        </p>
                      </li>
                    ))}
                  </Grid>
                </ul>
              </TableCell>
            ))}
        </div>

        <Button className="mt-2" onClick={() => handleSaveEmailSettings()}>
          Save Changes
        </Button>
        <Button
          onClick={async () => {
            if (!accessToken) return;
            try {
              // Silently attempt to persist the current form values so the
              // backend can read TEST_EMAIL_ADDRESS from the DB (DB mode).
              // If saving is not supported (e.g. STORE_MODEL_IN_DB=False /
              // YAML mode), this is a no-op and the backend will fall back to
              // the TEST_EMAIL_ADDRESS environment variable instead.
              await handleSaveEmailSettings({ silent: true });
              await serviceHealthCheck(accessToken, "email");
              NotificationManager.success("Email test triggered. Check your configured email inbox/logs.");
            } catch (error) {
              NotificationManager.fromBackend(error);
            }
          }}
          className="mx-2"
        >
          Test Email Alerts
        </Button>
      </Card>
    </>
  );
};

export default EmailSettings;
