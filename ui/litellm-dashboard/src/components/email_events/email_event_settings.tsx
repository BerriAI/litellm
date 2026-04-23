import React, { useState, useEffect } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Separator } from "@/components/ui/separator";
import { Skeleton } from "@/components/ui/skeleton";
import NotificationsManager from "../molecules/notifications_manager";
import {
  getEmailEventSettings,
  updateEmailEventSettings,
  resetEmailEventSettings,
} from "../networking";
import { EmailEvent } from "../../types";
import { EmailEventSetting } from "./types";

interface EmailEventSettingsProps {
  accessToken: string | null;
}

const EmailEventSettings: React.FC<EmailEventSettingsProps> = ({
  accessToken,
}) => {
  const [loading, setLoading] = useState(true);
  const [eventSettings, setEventSettings] = useState<EmailEventSetting[]>([]);

  useEffect(() => {
    const fetchEventSettings = async () => {
      if (!accessToken) return;
      setLoading(true);
      try {
        const response = await getEmailEventSettings(accessToken);
        setEventSettings(response.settings);
      } catch (error) {
        console.error("Failed to fetch email event settings:", error);
        NotificationsManager.fromBackend(error);
      } finally {
        setLoading(false);
      }
    };
    fetchEventSettings();
  }, [accessToken]);

  const handleCheckboxChange = (event: EmailEvent, checked: boolean) => {
    const updatedSettings = eventSettings.map((setting) =>
      setting.event === event ? { ...setting, enabled: checked } : setting,
    );
    setEventSettings(updatedSettings);
  };

  const handleSaveSettings = async () => {
    if (!accessToken) return;
    try {
      await updateEmailEventSettings(accessToken, { settings: eventSettings });
      NotificationsManager.success("Email event settings updated successfully");
    } catch (error) {
      console.error("Failed to update email event settings:", error);
      NotificationsManager.fromBackend(error);
    }
  };

  const handleResetSettings = async () => {
    if (!accessToken) return;
    try {
      await resetEmailEventSettings(accessToken);
      NotificationsManager.success("Email event settings reset to defaults");
      // Refresh
      setLoading(true);
      const response = await getEmailEventSettings(accessToken);
      setEventSettings(response.settings);
      setLoading(false);
    } catch (error) {
      console.error("Failed to reset email event settings:", error);
      NotificationsManager.fromBackend(error);
    }
  };

  const getEventDescription = (event: EmailEvent): string => {
    if (event.includes("Virtual Key Created")) {
      return "An email will be sent to the user when a new virtual key is created with their user ID";
    } else if (event.includes("New User Invitation")) {
      return "An email will be sent to the email address of the user when a new user is created";
    } else {
      const words = event
        .split(/(?=[A-Z])/)
        .join(" ")
        .toLowerCase();
      return `Receive an email notification when ${words}`;
    }
  };

  return (
    <Card className="p-6">
      <h4 className="text-lg font-semibold m-0">Email Notifications</h4>
      <p className="text-sm text-muted-foreground">
        Select which events should trigger email notifications.
      </p>
      <Separator className="my-4" />

      {loading ? (
        <div className="py-5 flex justify-center">
          <Skeleton className="h-8 w-48" />
        </div>
      ) : (
        <div className="space-y-4">
          {eventSettings.map((setting) => (
            <div key={setting.event} className="flex items-start gap-3">
              <Checkbox
                id={`email-event-${setting.event}`}
                checked={setting.enabled}
                onCheckedChange={(checked) =>
                  handleCheckboxChange(setting.event, !!checked)
                }
                className="mt-1"
              />
              <div>
                <label
                  htmlFor={`email-event-${setting.event}`}
                  className="text-sm font-medium cursor-pointer"
                >
                  {setting.event}
                </label>
                <div className="text-sm text-muted-foreground block">
                  {getEventDescription(setting.event)}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      <div className="mt-6 flex gap-3">
        <Button onClick={handleSaveSettings} disabled={loading}>
          Save Changes
        </Button>
        <Button
          onClick={handleResetSettings}
          variant="secondary"
          disabled={loading}
        >
          Reset to Defaults
        </Button>
      </div>
    </Card>
  );
};

export default EmailEventSettings;
