/**
 * UI for controlling slack alerting settings
 */
import React, { useState, useEffect } from "react";
import {
  Table,
  TableHead,
  TableRow,
  TableHeaderCell,
  TableCell,
  Button,
  Icon,
  Badge,
  TableBody,
  Text,
} from "@tremor/react";
import { InputNumber, message } from "antd";
import { alertingSettingsCall, updateConfigFieldSetting } from "../networking";
import { TrashIcon, CheckCircleIcon } from "@heroicons/react/outline";
import DynamicForm from "./dynamic_form";
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

const AlertingSettings: React.FC<AlertingSettingsProps> = ({
  accessToken,
  premiumUser,
}) => {
  const [alertingSettings, setAlertingSettings] = useState<
    alertingSettingsItem[]
  >([]);

  console.log("INSIDE ALERTING SETTINGS");
  useEffect(() => {
    // get values
    if (!accessToken) {
      return;
    }
    alertingSettingsCall(accessToken).then((data) => {
      setAlertingSettings(data);
    });
  }, [accessToken]);

  const handleInputChange = (fieldName: string, newValue: any) => {
    // Update the value in the state
    const updatedSettings = alertingSettings.map((setting) =>
      setting.field_name === fieldName
        ? { ...setting, field_value: newValue }
        : setting
    );
    setAlertingSettings(updatedSettings);
  };

  const handleSubmit = (formValues: Record<string, any>) => {
    if (!accessToken) {
      return;
    }

    let fieldValue = formValues;

    if (fieldValue == null || fieldValue == undefined) {
      return;
    }

    const initialFormValues: Record<string, any> = {};
    alertingSettings.forEach((setting) => {
      initialFormValues[setting.field_name] = setting.field_value;
    });

    // Merge initialFormValues with actual formValues
    const mergedFormValues = { ...formValues, ...initialFormValues };
    try {
      updateConfigFieldSetting(accessToken, "alerting_args", mergedFormValues);
      // update value in state
      message.success("Wait 10s for proxy to update.");
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
          : setting
      );
      console.log("INSIDE HANDLE RESET FIELD");
      setAlertingSettings(updatedSettings);
    } catch (error) {
      // do something
      console.log("ERROR OCCURRED!");
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
