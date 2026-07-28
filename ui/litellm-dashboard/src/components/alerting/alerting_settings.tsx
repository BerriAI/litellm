/**
 * UI for controlling slack alerting settings
 */
import React, { useState, useEffect, useRef } from "react";

import { alertingSettingsCall, updateConfigFieldSetting } from "../networking";
import DynamicForm from "./dynamic_form";
import NotificationsManager from "../molecules/notifications_manager";
interface alertingSettingsItem {
  field_name: string;
  field_type: string;
  field_value: any;
  field_default_value: any;
  field_description: string;
  stored_in_db: boolean | null;
  premium_field: boolean;
}

interface AlertingSettingsProps {
  accessToken: string | null;
  premiumUser: boolean;
}

const AlertingSettings: React.FC<AlertingSettingsProps> = ({ accessToken, premiumUser }) => {
  const [alertingSettings, setAlertingSettings] = useState<alertingSettingsItem[]>([]);
  // `general_settings.alerting` can hold channels this page does not manage (e.g. "email"),
  // so it is only rewritten when the slack_alerting switch itself changed.
  const initialSlackAlerting = useRef<boolean | null>(null);

  useEffect(() => {
    // get values
    if (!accessToken) {
      return;
    }
    alertingSettingsCall(accessToken).then((data) => {
      setAlertingSettings(data);
      const slackSetting = data.find((setting: alertingSettingsItem) => setting.field_name === "slack_alerting");
      initialSlackAlerting.current = typeof slackSetting?.field_value === "boolean" ? slackSetting.field_value : null;
    });
  }, [accessToken]);

  const handleInputChange = (fieldName: string, newValue: any) => {
    // Update the value in the state
    const updatedSettings = alertingSettings.map((setting) =>
      setting.field_name === fieldName ? { ...setting, field_value: newValue } : setting,
    );

    setAlertingSettings(updatedSettings);
  };

  const handleSubmit = (formValues: Record<string, any>) => {
    if (!accessToken) {
      return;
    }

    if (formValues == null || formValues == undefined) {
      return;
    }

    const { slack_alerting, ...alertingArgs } = formValues;
    try {
      updateConfigFieldSetting(accessToken, "alerting_args", alertingArgs);
      if (typeof slack_alerting === "boolean" && slack_alerting !== initialSlackAlerting.current) {
        updateConfigFieldSetting(accessToken, "alerting", slack_alerting ? ["slack"] : []);
        initialSlackAlerting.current = slack_alerting;
      }
      // update value in state
      NotificationsManager.success("Wait 10s for proxy to update.");
    } catch (error) {
      // do something
    }
  };

  const handleResetField = (fieldName: string, idx: number) => {
    if (!accessToken) {
      return;
    }

    try {
      //   deleteConfigFieldSetting(accessToken, fieldName);
      // update value in state

      const updatedSettings = alertingSettings.map((setting) =>
        setting.field_name === fieldName
          ? {
              ...setting,
              stored_in_db: null,
              field_value: setting.field_default_value,
            }
          : setting,
      );
      setAlertingSettings(updatedSettings);
    } catch (error) {
      // do something
    }
  };

  return (
    <DynamicForm
      alertingSettings={alertingSettings}
      handleInputChange={handleInputChange}
      handleResetField={handleResetField}
      handleSubmit={handleSubmit}
      premiumUser={premiumUser}
    />
  );
};

export default AlertingSettings;
