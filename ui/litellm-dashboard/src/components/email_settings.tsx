import React from "react";
import { Card, Text, Grid, Button, TextInput, TableCell } from "@tremor/react";
import { Typography } from "antd";
import { useTranslation } from "react-i18next";
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
  const { t } = useTranslation();
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
      NotificationManager.success(t("emailSettings.saveSuccess"));
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
        <Title level={4}>{t("emailSettings.emailServerSettings")}</Title>
        <Text>
          <a href="https://docs.litellm.ai/docs/proxy/email" target="_blank" style={{ color: "blue" }}>
            {" "}
            {t("emailSettings.docsLink")}
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
                              {t("emailSettings.smtpHostDesc")}
                              <span style={{ color: "red" }}> {t("emailSettings.requiredStar")} </span>
                            </div>
                          )}

                          {key === "SMTP_PORT" && (
                            <div style={{ color: "gray" }}>
                              {t("emailSettings.smtpPortDesc")}
                              <span style={{ color: "red" }}> {t("emailSettings.requiredStar")} </span>
                            </div>
                          )}

                          {key === "SMTP_USERNAME" && (
                            <div style={{ color: "gray" }}>
                              {t("emailSettings.smtpUsernameDesc")}
                              <span style={{ color: "red" }}> {t("emailSettings.requiredStar")} </span>
                            </div>
                          )}

                          {key === "SMTP_PASSWORD" && (
                            <span style={{ color: "red" }}> {t("emailSettings.requiredStar")} </span>
                          )}

                          {key === "SMTP_SENDER_EMAIL" && (
                            <div style={{ color: "gray" }}>
                              {t("emailSettings.smtpSenderEmailDesc")}
                              <span style={{ color: "red" }}> {t("emailSettings.requiredStar")} </span>
                            </div>
                          )}

                          {key === "TEST_EMAIL_ADDRESS" && (
                            <div style={{ color: "gray" }}>
                              {t("emailSettings.testEmailAddressDesc")}
                              <span style={{ color: "red" }}> {t("emailSettings.requiredStar")} </span>
                            </div>
                          )}
                          {key === "EMAIL_LOGO_URL" && (
                            <div style={{ color: "gray" }}>{t("emailSettings.emailLogoUrlDesc")}</div>
                          )}
                          {key === "EMAIL_SUPPORT_CONTACT" && (
                            <div style={{ color: "gray" }}>{t("emailSettings.emailSupportContactDesc")}</div>
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
          {t("emailSettings.saveChanges")}
        </Button>
        <Button
          onClick={async () => {
            if (!accessToken) return;
            try {
              await serviceHealthCheck(accessToken, "email");
              NotificationManager.success(t("emailSettings.testSuccess"));
            } catch (error) {
              NotificationManager.fromBackend(error);
            }
          }}
          className="mx-2"
        >
          {t("emailSettings.testEmailAlerts")}
        </Button>
      </Card>
    </>
  );
};

export default EmailSettings;
