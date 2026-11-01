import React, { useState } from "react";
import { Trans, useTranslation } from "react-i18next";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { InputGroup, InputGroupAddon, InputGroupButton, InputGroupInput } from "@/components/ui/input-group";
import { Eye, EyeOff } from "lucide-react";
import { toast } from "@/lib/toast";
import { serviceHealthCheck, setCallbacksCall } from "./networking";
import { EmailEventSettings } from "./email_events";

interface EmailSettingsProps {
  accessToken: string | null;
  premiumUser: boolean;
  alerts: any[];
}

const REQUIRED_MARKER = (
  <span className="text-destructive">
    <Trans i18nKey="loggingAndAlerts.emailSettings.required" />
  </span>
);

const FIELD_HELP: Record<string, React.ReactNode> = {
  SMTP_HOST: (
    <>
      <Trans i18nKey="loggingAndAlerts.emailSettings.help.smtpHost" />
      {REQUIRED_MARKER}
    </>
  ),
  SMTP_PORT: (
    <>
      <Trans i18nKey="loggingAndAlerts.emailSettings.help.smtpPort" />
      {REQUIRED_MARKER}
    </>
  ),
  SMTP_USERNAME: (
    <>
      <Trans i18nKey="loggingAndAlerts.emailSettings.help.smtpUsername" />
      {REQUIRED_MARKER}
    </>
  ),
  SMTP_PASSWORD: REQUIRED_MARKER,
  SMTP_SENDER_EMAIL: (
    <>
      <Trans i18nKey="loggingAndAlerts.emailSettings.help.smtpSenderEmail" />
      {REQUIRED_MARKER}
    </>
  ),
  TEST_EMAIL_ADDRESS: (
    <>
      <Trans i18nKey="loggingAndAlerts.emailSettings.help.testEmailAddress" />
      {REQUIRED_MARKER}
    </>
  ),
  EMAIL_LOGO_URL: <Trans i18nKey="loggingAndAlerts.emailSettings.help.emailLogoUrl" />,
  EMAIL_SUPPORT_CONTACT: <Trans i18nKey="loggingAndAlerts.emailSettings.help.emailSupportContact" />,
};

const PREMIUM_ONLY_FIELDS = ["EMAIL_LOGO_URL", "EMAIL_SUPPORT_CONTACT"];

const SENSITIVE_FIELD_PATTERN = /(PASSWORD|SECRET|KEY|TOKEN)/i;

const EmailSettings: React.FC<EmailSettingsProps> = ({ accessToken, premiumUser, alerts }) => {
  const { t } = useTranslation();
  const [visibleFields, setVisibleFields] = useState<Record<string, boolean>>({});

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
      toast.success(t("loggingAndAlerts.emailSettings.toasts.saved"));
    } catch (error) {
      toast.fromError(error);
    }
  };

  return (
    <>
      <div className="mt-6 mb-6">
        <EmailEventSettings accessToken={accessToken} />
      </div>
      <Card>
        <CardHeader>
          <CardTitle className="text-base">{t("loggingAndAlerts.emailSettings.title")}</CardTitle>
          <p className="text-sm">
            <a
              href="https://docs.litellm.ai/docs/proxy/email"
              target="_blank"
              rel="noreferrer"
              className="text-primary underline underline-offset-4"
            >
              {t("loggingAndAlerts.emailSettings.docsLink")}
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
                          className="text-sm text-primary underline underline-offset-4"
                        >
                          ✨ {key}
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
                              aria-label={
                                isVisible
                                  ? t("loggingAndAlerts.emailSettings.hideCredential")
                                  : t("loggingAndAlerts.emailSettings.showCredential")
                              }
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
            <Button onClick={() => handleSaveEmailSettings()}>{t("loggingAndAlerts.emailSettings.saveChanges")}</Button>
            <Button
              variant="secondary"
              onClick={async () => {
                if (!accessToken) return;
                try {
                  await serviceHealthCheck(accessToken, "email");
                  toast.success(t("loggingAndAlerts.emailSettings.toasts.testTriggered"));
                } catch (error) {
                  toast.fromError(error);
                }
              }}
            >
              {t("loggingAndAlerts.emailSettings.testEmailAlerts")}
            </Button>
          </div>
        </CardContent>
      </Card>
    </>
  );
};

export default EmailSettings;
