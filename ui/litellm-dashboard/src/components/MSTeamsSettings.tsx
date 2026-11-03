import React, { useState } from "react";
import { Trans, useTranslation } from "react-i18next";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { InputGroup, InputGroupAddon, InputGroupButton, InputGroupInput } from "@/components/ui/input-group";
import { Eye, EyeOff } from "lucide-react";
import { toast } from "@/lib/toast";
import { getCallbacksCall, serviceHealthCheck, setCallbacksCall } from "./networking";

interface AlertingDestination {
  name: string;
  variables?: Record<string, string | null>;
}

interface MSTeamsSettingsProps {
  accessToken: string | null;
  userID: string | null;
  userRole: string | null;
  alerts: AlertingDestination[];
}

const FIELD_HELP: Record<string, React.ReactNode> = {
  MS_TEAMS_WEBHOOK_URL: (
    <>
      <Trans i18nKey="loggingAndAlerts.msTeams.help.webhookUrl" />
      <span className="text-destructive">
        <Trans i18nKey="loggingAndAlerts.msTeams.required" />
      </span>
    </>
  ),
};

const SENSITIVE_FIELD_PATTERN = /(PASSWORD|SECRET|KEY|TOKEN|URL)/i;

const MSTeamsSettings: React.FC<MSTeamsSettingsProps> = ({ accessToken, userID, userRole, alerts }) => {
  const { t } = useTranslation();
  const [visibleFields, setVisibleFields] = useState<Record<string, boolean>>({});

  const toggleFieldVisibility = (key: string) => {
    setVisibleFields((prev) => ({
      ...prev,
      [key]: !prev[key],
    }));
  };

  const handleSaveMSTeamsSettings = async () => {
    if (!accessToken || !userID || !userRole) {
      return;
    }

    // Only send fields the admin actually edited. Values rendered from the
    // server are masked or sourced from the process environment, so
    // re-submitting an untouched field would persist a mask or copy
    // env-managed config into the database.
    const updatedVariables: Record<string, string> = Object.fromEntries(
      alerts
        .filter((alert) => alert.name === "ms_teams")
        .flatMap((alert) =>
          Object.entries(alert.variables ?? {}).flatMap(([key, value]) => {
            const inputElement = document.querySelector(`input[name="${key}"]`) as HTMLInputElement;
            if (!inputElement || !inputElement.value) {
              return [];
            }
            if (inputElement.value === (value == null ? "" : String(value))) {
              return [];
            }
            return [[key, inputElement.value] as const];
          }),
        ),
    );

    try {
      // Re-read the persisted destinations at save time so that a Teams save
      // never restores destinations another form disabled after page load.
      const currentConfig = await getCallbacksCall(accessToken, userID, userRole);
      const currentDestinations: string[] = currentConfig.active_alerting_destinations ?? [];
      const payload = {
        general_settings: {
          alerting: Array.from(new Set([...currentDestinations, "ms_teams"])),
        },
        environment_variables: updatedVariables,
      };
      await setCallbacksCall(accessToken, payload);
      toast.success(t("loggingAndAlerts.msTeams.toasts.saved"));
    } catch (error) {
      toast.fromError(error);
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">{t("loggingAndAlerts.msTeams.title")}</CardTitle>
        <p className="text-sm">
          <Trans
            i18nKey="loggingAndAlerts.msTeams.intro"
            components={{
              docs: (
                <a
                  href="https://learn.microsoft.com/en-us/microsoftteams/platform/webhooks-and-connectors/how-to/add-incoming-webhook"
                  target="_blank"
                  rel="noreferrer"
                  className="text-primary underline underline-offset-4"
                />
              ),
            }}
          />
        </p>
      </CardHeader>

      <CardContent>
        {alerts
          .filter((alert) => alert.name === "ms_teams")
          .map((alert, index) => (
            <div key={index} className="grid grid-cols-1 gap-4 sm:grid-cols-2">
              {Object.entries(alert.variables ?? {}).map(([key, value]) => {
                const isSensitive = SENSITIVE_FIELD_PATTERN.test(key);
                const isVisible = visibleFields[key] || false;
                return (
                  <div key={key} className="space-y-1">
                    <p className="text-sm">{key}</p>
                    <InputGroup className="max-w-100">
                      <InputGroupInput
                        name={key}
                        defaultValue={value as string}
                        type={isSensitive && !isVisible ? "password" : "text"}
                      />
                      {isSensitive && (
                        <InputGroupAddon align="inline-end">
                          <InputGroupButton
                            size="icon-xs"
                            onClick={() => toggleFieldVisibility(key)}
                            aria-label={
                              isVisible
                                ? t("loggingAndAlerts.msTeams.hideCredential")
                                : t("loggingAndAlerts.msTeams.showCredential")
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
          <Button onClick={() => handleSaveMSTeamsSettings()}>{t("loggingAndAlerts.msTeams.saveChanges")}</Button>
          <Button
            variant="secondary"
            onClick={async () => {
              if (!accessToken) return;
              try {
                await serviceHealthCheck(accessToken, "ms_teams");
                toast.success(t("loggingAndAlerts.msTeams.toasts.testTriggered"));
              } catch (error) {
                toast.fromError(error);
              }
            }}
          >
            {t("loggingAndAlerts.msTeams.testAlerts")}
          </Button>
        </div>
      </CardContent>
    </Card>
  );
};

export default MSTeamsSettings;
