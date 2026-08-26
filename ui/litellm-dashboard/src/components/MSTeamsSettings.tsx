import React, { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { InputGroup, InputGroupAddon, InputGroupButton, InputGroupInput } from "@/components/ui/input-group";
import { Eye, EyeOff } from "lucide-react";
import { toast } from "@/lib/toast";
import { serviceHealthCheck, setCallbacksCall } from "./networking";

interface AlertingDestination {
  name: string;
  variables?: Record<string, string | null>;
}

interface MSTeamsSettingsProps {
  accessToken: string | null;
  alerts: AlertingDestination[];
  activeAlertingDestinations: string[];
}

const FIELD_HELP: Record<string, React.ReactNode> = {
  MS_TEAMS_WEBHOOK_URL: (
    <>
      Incoming webhook URL for your Teams channel (Workflows or incoming webhook connector)
      <span className="text-destructive"> Required * </span>
    </>
  ),
};

const SENSITIVE_FIELD_PATTERN = /(PASSWORD|SECRET|KEY|TOKEN|URL)/i;

const MSTeamsSettings: React.FC<MSTeamsSettingsProps> = ({ accessToken, alerts, activeAlertingDestinations }) => {
  const [visibleFields, setVisibleFields] = useState<Record<string, boolean>>({});

  const toggleFieldVisibility = (key: string) => {
    setVisibleFields((prev) => ({
      ...prev,
      [key]: !prev[key],
    }));
  };

  const handleSaveMSTeamsSettings = async () => {
    if (!accessToken) {
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

    const payload = {
      general_settings: {
        alerting: Array.from(new Set([...activeAlertingDestinations, "ms_teams"])),
      },
      environment_variables: updatedVariables,
    };
    try {
      await setCallbacksCall(accessToken, payload);
      toast.success("MS Teams settings updated successfully");
    } catch (error) {
      toast.fromError(error);
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Microsoft Teams Alerting Settings</CardTitle>
        <p className="text-sm">
          Send LiteLLM alerts to a Microsoft Teams channel via an incoming webhook. Create one from{" "}
          <a
            href="https://learn.microsoft.com/en-us/microsoftteams/platform/webhooks-and-connectors/how-to/add-incoming-webhook"
            target="_blank"
            rel="noreferrer"
            className="text-primary underline underline-offset-4"
          >
            Microsoft Docs: incoming webhooks
          </a>
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
          <Button onClick={() => handleSaveMSTeamsSettings()}>Save Changes</Button>
          <Button
            variant="secondary"
            onClick={async () => {
              if (!accessToken) return;
              try {
                await serviceHealthCheck(accessToken, "ms_teams");
                toast.success("MS Teams test alert triggered. Check your Teams channel.");
              } catch (error) {
                toast.fromError(error);
              }
            }}
          >
            Test MS Teams Alerts
          </Button>
        </div>
      </CardContent>
    </Card>
  );
};

export default MSTeamsSettings;
