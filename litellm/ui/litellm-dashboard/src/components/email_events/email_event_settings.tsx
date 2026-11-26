import React, { useEffect, useId, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { Separator } from "@/components/ui/separator";
import { Skeleton } from "@/components/ui/skeleton";
import { toast } from "@/lib/toast";
import { getEmailEventSettings, updateEmailEventSettings, resetEmailEventSettings } from "../networking";
import { EmailEvent } from "../../types";
import { EmailEventSetting } from "./types";

interface EmailEventSettingsProps {
  accessToken: string | null;
}

const EmailEventSettings: React.FC<EmailEventSettingsProps> = ({ accessToken }) => {
  const { t } = useTranslation();
  const [loading, setLoading] = useState(true);
  const [eventSettings, setEventSettings] = useState<EmailEventSetting[]>([]);
  const [savedRecipient, setSavedRecipient] = useState("");
  const [recipient, setRecipient] = useState("");
  const recipientId = useId();
  const recipientInput = useRef<HTMLInputElement>(null);

  // Fetch email event settings on component mount
  useEffect(() => {
    fetchEventSettings();
  }, [accessToken]);

  const fetchEventSettings = async () => {
    if (!accessToken) return;

    setLoading(true);
    try {
      const response = await getEmailEventSettings(accessToken);
      setEventSettings(response.settings);
      setSavedRecipient(response.model_deprecation_email_recipient ?? "");
      setRecipient(response.model_deprecation_email_recipient ?? "");
    } catch (error) {
      console.error("Failed to fetch email event settings:", error);
      toast.fromError(error);
    } finally {
      setLoading(false);
    }
  };

  const handleEnabledChange = (event: EmailEvent, checked: boolean) => {
    const updatedSettings = eventSettings.map((setting) =>
      setting.event === event ? { ...setting, enabled: checked } : setting,
    );
    setEventSettings(updatedSettings);
  };

  const handleSaveSettings = async () => {
    if (!accessToken) return;

    try {
      const address = recipient.trim();
      const changed = address !== savedRecipient;
      if (changed && !recipientInput.current?.reportValidity()) return;

      await updateEmailEventSettings(accessToken, {
        settings: eventSettings,
        ...(changed ? { model_deprecation_email_recipient: address || null } : {}),
      });
      if (changed) setSavedRecipient(address);
      toast.success(t("loggingAndAlerts.emailEvents.toasts.saved"));
    } catch (error) {
      console.error("Failed to update email event settings:", error);
      toast.fromError(error);
    }
  };

  const handleResetSettings = async () => {
    if (!accessToken) return;

    try {
      await resetEmailEventSettings(accessToken);
      toast.success(t("loggingAndAlerts.emailEvents.toasts.reset"));
      // Refresh settings after reset
      fetchEventSettings();
    } catch (error) {
      console.error("Failed to reset email event settings:", error);
      toast.fromError(error);
    }
  };

  // Helper function to get a description for each event type
  const getEventDescription = (event: EmailEvent): string => {
    if (event === "Performance Alerts") {
      return t("loggingAndAlerts.emailEvents.descriptions.performanceAlerts");
    }
    if (event === "Model Deprecations") {
      return t("loggingAndAlerts.emailEvents.descriptions.modelDeprecations");
    }
    // Convert event name to a sentence with more context
    if (event.includes("Virtual Key Created")) {
      return t("loggingAndAlerts.emailEvents.descriptions.virtualKeyCreated");
    } else if (event.includes("New User Invitation")) {
      return t("loggingAndAlerts.emailEvents.descriptions.newUserInvitation");
    } else {
      // Handle any other event type from the API
      const words = event
        .split(/(?=[A-Z])/)
        .join(" ")
        .toLowerCase();
      return t("loggingAndAlerts.emailEvents.descriptions.generic", { event: words });
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">{t("loggingAndAlerts.emailEvents.title")}</CardTitle>
        <p className="text-sm text-muted-foreground">{t("loggingAndAlerts.emailEvents.description")}</p>
      </CardHeader>

      <CardContent>
        <Separator className="mb-6" />

        {loading ? (
          <div className="space-y-4">
            <Skeleton className="h-10 w-full" />
            <Skeleton className="h-10 w-full" />
          </div>
        ) : (
          <div className="space-y-4">
            {eventSettings.map((setting) => (
              <div key={setting.event} className="flex items-start">
                <Switch
                  checked={setting.enabled}
                  onCheckedChange={(checked) => handleEnabledChange(setting.event, checked)}
                  aria-label={setting.event}
                  className="mt-1"
                />
                <div className="ml-3">
                  <p className="text-sm">{setting.event}</p>
                  <div className="block text-sm text-muted-foreground">{getEventDescription(setting.event)}</div>
                  {setting.event === "Model Deprecations" && (
                    <div className="mt-3 max-w-sm space-y-1">
                      <label className="text-sm font-medium" htmlFor={recipientId}>
                        {t("loggingAndAlerts.emailEvents.deprecationRecipient.label")}
                      </label>
                      <Input
                        id={recipientId}
                        ref={recipientInput}
                        type="email"
                        value={recipient}
                        onChange={(event) => setRecipient(event.target.value)}
                        aria-describedby={`${recipientId}-hint`}
                      />
                      <p id={`${recipientId}-hint`} className="text-sm text-muted-foreground">
                        {t("loggingAndAlerts.emailEvents.deprecationRecipient.hint")}
                      </p>
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}

        <div className="mt-6 flex gap-4">
          <Button onClick={handleSaveSettings} disabled={loading}>
            {t("loggingAndAlerts.emailEvents.saveChanges")}
          </Button>
          <Button variant="secondary" onClick={handleResetSettings} disabled={loading}>
            {t("loggingAndAlerts.emailEvents.resetToDefaults")}
          </Button>
        </div>
      </CardContent>
    </Card>
  );
};

export default EmailEventSettings;
