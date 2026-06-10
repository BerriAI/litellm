import React, { useState, useEffect } from "react";
import { useTranslation } from "react-i18next";
import { Card, Text, Button } from "@tremor/react";
import { Typography, Divider, Spin, Checkbox } from "antd";
import NotificationsManager from "../molecules/notifications_manager";
import { getEmailEventSettings, updateEmailEventSettings, resetEmailEventSettings } from "../networking";
import { EmailEvent } from "../../types";
import { EmailEventSetting } from "./types";
import { TFunction } from "i18next";

const { Title } = Typography;

interface EmailEventSettingsProps {
  accessToken: string | null;
}

const EmailEventSettings: React.FC<EmailEventSettingsProps> = ({ accessToken }) => {
  const { t } = useTranslation();
  const [loading, setLoading] = useState(true);
  const [eventSettings, setEventSettings] = useState<EmailEventSetting[]>([]);

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
    } catch (error) {
      console.error("Failed to fetch email event settings:", error);
      NotificationsManager.fromBackend(error);
    } finally {
      setLoading(false);
    }
  };

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
      NotificationsManager.success(t("emailEvents.emailEventSettings.saveSuccess"));
    } catch (error) {
      console.error("Failed to update email event settings:", error);
      NotificationsManager.fromBackend(error);
    }
  };

  const handleResetSettings = async () => {
    if (!accessToken) return;

    try {
      await resetEmailEventSettings(accessToken);
      NotificationsManager.success(t("emailEvents.emailEventSettings.resetSuccess"));
      // Refresh settings after reset
      fetchEventSettings();
    } catch (error) {
      console.error("Failed to reset email event settings:", error);
      NotificationsManager.fromBackend(error);
    }
  };

  // Helper function to get a description for each event type
  const getEventDescription = (event: EmailEvent, tFn: TFunction): string => {
    // Convert event name to a sentence with more context
    if (event.includes("Virtual Key Created")) {
      return tFn("emailEvents.emailEventSettings.descVirtualKeyCreated");
    } else if (event.includes("New User Invitation")) {
      return tFn("emailEvents.emailEventSettings.descNewUserInvitation");
    } else {
      // Handle any other event type from the API
      const words = event
        .split(/(?=[A-Z])/)
        .join(" ")
        .toLowerCase();
      return tFn("emailEvents.emailEventSettings.descGeneric", { words });
    }
  };

  return (
    <Card>
      <Title level={4}>{t("emailEvents.emailEventSettings.title")}</Title>
      <Text>{t("emailEvents.emailEventSettings.subtitle")}</Text>
      <Divider />

      {loading ? (
        <div style={{ textAlign: "center", padding: "20px" }}>
          <Spin size="large" />
        </div>
      ) : (
        <div className="space-y-4">
          {eventSettings.map((setting) => (
            <div key={setting.event} className="flex items-center">
              <Checkbox
                checked={setting.enabled}
                onChange={(e) => handleCheckboxChange(setting.event, e.target.checked)}
              />
              <div className="ml-3">
                <Text>{setting.event}</Text>
                <div className="text-sm text-gray-500 block">{getEventDescription(setting.event, t)}</div>
              </div>
            </div>
          ))}
        </div>
      )}

      <div className="mt-6 flex space-x-4">
        <Button onClick={handleSaveSettings} disabled={loading}>
          {t("emailEvents.emailEventSettings.saveChanges")}
        </Button>
        <Button onClick={handleResetSettings} variant="secondary" disabled={loading}>
          {t("emailEvents.emailEventSettings.resetToDefaults")}
        </Button>
      </div>
    </Card>
  );
};

export default EmailEventSettings;
