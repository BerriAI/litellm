/**
 * UI for controlling slack alerting settings
 */
import React, { useState, useEffect } from "react";

import { alertingSettingsCall, updateConfigFieldSetting } from "../networking";
import DynamicForm, { type AlertingFieldValue, type AlertingFormValues } from "./dynamic_form";
import { toast } from "@/lib/toast";
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

  useEffect(() => {
    // get values
    if (!accessToken) {
      return;
    }
    alertingSettingsCall(accessToken).then((data) => {
      setAlertingSettings(data);
    });
  }, [accessToken]);

  const handleInputChange = (fieldName: string, newValue: AlertingFieldValue) => {
    // Update the value in the state
    const updatedSettings = alertingSettings.map((setting) =>
      setting.field_name === fieldName ? { ...setting, field_value: newValue } : setting,
    );

    setAlertingSettings(updatedSettings);
  };

  const handleSubmit = async (formValues: AlertingFormValues) => {
    if (!accessToken) {
      return;
    }

    const storedValues = Object.fromEntries(
      alertingSettings.map((setting) => [setting.field_name, setting.field_value]),
    );
    const { slack_alerting, ...editedArgs } = { ...formValues, ...storedValues };
    const alertingArgs = Object.fromEntries(
      Object.entries(editedArgs).filter(([, value]) => value !== null && value !== undefined && value !== ""),
    );

    try {
      await updateConfigFieldSetting(accessToken, "alerting_args", alertingArgs);
      if (typeof slack_alerting === "boolean") {
        await updateConfigFieldSetting(accessToken, "alerting", slack_alerting ? ["slack"] : []);
      }
      toast.success("Wait 10s for proxy to update.");
    } catch (error) {
      toast.fromError(error);
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
