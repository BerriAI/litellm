/**
 * UI for controlling slack alerting settings
 */
import React, { useState, useEffect } from "react";

import { alertingSettingsCall, updateConfigFieldSetting } from "../networking";
import DynamicForm from "./dynamic_form";
import { extractProxyErrorMessage } from "@/lib/http/client";
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
  const [resetFields, setResetFields] = useState<Set<string>>(new Set());

  useEffect(() => {
    if (!accessToken) {
      return;
    }

    const controller = new AbortController();
    alertingSettingsCall(accessToken).then((data) => {
      if (controller.signal.aborted) {
        return;
      }
      setResetFields(new Set());
      setAlertingSettings(data);
    });

    return () => {
      controller.abort();
    };
  }, [accessToken]);

  const handleInputChange = (fieldName: string, newValue: any) => {
    setResetFields((previous) =>
      previous.has(fieldName) ? new Set([...previous].filter((name) => name !== fieldName)) : previous,
    );

    const updatedSettings = alertingSettings.map((setting) =>
      setting.field_name === fieldName ? { ...setting, field_value: newValue } : setting,
    );

    setAlertingSettings(updatedSettings);
  };

  const handleSubmit = async (formValues: Record<string, any>) => {
    if (!accessToken) {
      return;
    }

    const fieldValue = formValues;

    if (fieldValue == null || fieldValue == undefined) {
      return;
    }

    const configuredAlertingArgs: Record<string, unknown> = Object.fromEntries(
      alertingSettings
        .filter(
          (setting) =>
            setting.field_name !== "slack_alerting" &&
            !resetFields.has(setting.field_name) &&
            setting.field_value != null,
        )
        .map((setting) => [setting.field_name, setting.field_value]),
    );

    const { slack_alerting, ...updatedAlertingArgs } = formValues;
    const alertingArgs = {
      ...configuredAlertingArgs,
      ...Object.fromEntries(
        Object.entries(updatedAlertingArgs).filter(
          ([fieldName, value]) => !resetFields.has(fieldName) && value != null && value !== "",
        ),
      ),
    };
    try {
      await updateConfigFieldSetting(accessToken, "alerting_args", alertingArgs);
      if (typeof slack_alerting === "boolean") {
        if (slack_alerting == true) {
          await updateConfigFieldSetting(accessToken, "alerting", ["slack"]);
        } else {
          await updateConfigFieldSetting(accessToken, "alerting", []);
        }
      }
      // update value in state
      toast.success("Wait 10s for proxy to update.");
    } catch (error) {
      toast.error(extractProxyErrorMessage(error));
    }
  };

  const handleResetField = (fieldName: string, idx: number) => {
    if (!accessToken) {
      return;
    }

    try {
      setResetFields((previous) => new Set([...previous, fieldName]));

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
      key={accessToken ?? "no-access-token"}
      alertingSettings={alertingSettings}
      handleInputChange={handleInputChange}
      handleResetField={handleResetField}
      handleSubmit={handleSubmit}
      premiumUser={premiumUser}
    />
  );
};

export default AlertingSettings;
