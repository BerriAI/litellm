import React, { useState, useEffect } from "react";
import {
  Card,
  Table,
  TableHead,
  TableRow,
  TableHeaderCell,
  TableCell,
  TableBody,
  Text,
  Button,
  Icon,
  Switch,
} from "@tremor/react";
import { TabPanel, TabPanels, TabGroup, TabList, Tab } from "@tremor/react";
import { getGeneralSettingsCall, updateConfigFieldSetting, deleteConfigFieldSetting } from "@/components/networking";
import { InputNumber, Select as AntdSelect } from "antd";
import { TrashIcon } from "@heroicons/react/outline";
import { StatusBadge } from "@/components/shared/table_cells";

import RouterSettings from "@/components/router_settings";
import Fallbacks from "@/components/Settings/RouterSettings/Fallbacks/Fallbacks";
import RoutingGroups from "@/components/routing_groups";
interface GeneralSettingsPageProps {
  accessToken: string | null;
  userRole: string | null;
  userID: string | null;
}

interface generalSettingsItem {
  field_name: string;
  field_type: string;
  field_value: any;
  field_description: string;
  stored_in_db: boolean | null;
  field_options?: string[] | null;
}

const SettingValueEditor: React.FC<{
  setting: generalSettingsItem;
  onChange: (fieldName: string, newValue: any) => void;
}> = ({ setting, onChange }) => {
  if (setting.field_type === "Integer") {
    return (
      <InputNumber
        step={1}
        value={setting.field_value}
        onChange={(newValue) => onChange(setting.field_name, newValue)}
      />
    );
  }
  if (setting.field_type === "Boolean") {
    return (
      <Switch
        checked={setting.field_value === true || setting.field_value === "true"}
        onChange={(checked) => onChange(setting.field_name, checked)}
      />
    );
  }
  if (setting.field_type === "Float") {
    return (
      <InputNumber
        min={0}
        max={1}
        step={0.05}
        value={setting.field_value}
        onChange={(newValue) => onChange(setting.field_name, newValue)}
      />
    );
  }
  if (setting.field_type === "Select") {
    return (
      <AntdSelect
        allowClear
        style={{ minWidth: "8rem" }}
        placeholder="Default"
        value={setting.field_value || undefined}
        options={(setting.field_options ?? []).map((option) => ({ label: option, value: option }))}
        onChange={(newValue) => onChange(setting.field_name, newValue ?? "")}
      />
    );
  }
  return null;
};

const GeneralSettings: React.FC<GeneralSettingsPageProps> = ({ accessToken, userRole, userID }) => {
  const [generalSettings, setGeneralSettings] = useState<generalSettingsItem[]>([]);

  useEffect(() => {
    if (!accessToken) {
      return;
    }
    getGeneralSettingsCall(accessToken).then((data) => {
      let general_settings = data;
      setGeneralSettings(general_settings);
    });
  }, [accessToken]);

  const handleInputChange = (fieldName: string, newValue: any) => {
    // Update the value in the state
    const updatedSettings = generalSettings.map((setting) =>
      setting.field_name === fieldName ? { ...setting, field_value: newValue } : setting,
    );
    setGeneralSettings(updatedSettings);
  };

  const handleUpdateField = (fieldName: string, idx: number) => {
    if (!accessToken) {
      return;
    }

    let fieldValue = generalSettings[idx].field_value;

    if (fieldValue == null || fieldValue == undefined) {
      return;
    }
    try {
      updateConfigFieldSetting(accessToken, fieldName, fieldValue);
      // update value in state

      const updatedSettings = generalSettings.map((setting) =>
        setting.field_name === fieldName ? { ...setting, stored_in_db: true } : setting,
      );
      setGeneralSettings(updatedSettings);
    } catch (error) {
      // do something
    }
  };

  const handleResetField = (fieldName: string, idx: number) => {
    if (!accessToken) {
      return;
    }

    try {
      deleteConfigFieldSetting(accessToken, fieldName);
      // update value in state

      const updatedSettings = generalSettings.map((setting) =>
        setting.field_name === fieldName ? { ...setting, stored_in_db: null, field_value: null } : setting,
      );
      setGeneralSettings(updatedSettings);
    } catch (error) {
      // do something
    }
  };

  if (!accessToken) {
    return null;
  }

  return (
    <div className="w-full">
      <TabGroup className="h-[75vh] w-full">
        <TabList variant="line" defaultValue="1" className="px-8 pt-4">
          <Tab value="1">Loadbalancing</Tab>
          <Tab value="2">Routing Groups</Tab>
          <Tab value="3">Fallbacks</Tab>
          <Tab value="4">General</Tab>
        </TabList>
        <TabPanels className="px-8 py-6">
          <TabPanel>
            <RouterSettings accessToken={accessToken} userRole={userRole} userID={userID} />
          </TabPanel>
          <TabPanel>
            <RoutingGroups />
          </TabPanel>
          <TabPanel>
            <Fallbacks accessToken={accessToken} userRole={userRole} userID={userID} />
          </TabPanel>
          <TabPanel>
            <Card>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableHeaderCell>Setting</TableHeaderCell>
                    <TableHeaderCell>Value</TableHeaderCell>
                    <TableHeaderCell>Status</TableHeaderCell>
                    <TableHeaderCell>Action</TableHeaderCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {generalSettings
                    .filter((value) => value.field_type !== "TypedDictionary")
                    .map((value, index) => (
                      <TableRow key={index}>
                        <TableCell>
                          <Text>{value.field_name}</Text>
                          <p
                            style={{
                              fontSize: "0.65rem",
                              color: "#808080",
                              fontStyle: "italic",
                            }}
                            className="mt-1"
                          >
                            {value.field_description}
                          </p>
                        </TableCell>
                        <TableCell>
                          <SettingValueEditor setting={value} onChange={handleInputChange} />
                        </TableCell>
                        <TableCell>
                          {value.stored_in_db == true ? (
                            <StatusBadge tone="success" label="In DB" />
                          ) : value.stored_in_db == false ? (
                            <StatusBadge tone="neutral" label="In Config" />
                          ) : (
                            <StatusBadge tone="neutral" label="Not Set" />
                          )}
                        </TableCell>
                        <TableCell>
                          <Button onClick={() => handleUpdateField(value.field_name, index)}>Update</Button>
                          <Icon icon={TrashIcon} color="red" onClick={() => handleResetField(value.field_name, index)}>
                            Reset
                          </Icon>
                        </TableCell>
                      </TableRow>
                    ))}
                </TableBody>
              </Table>
            </Card>
          </TabPanel>
        </TabPanels>
      </TabGroup>
    </div>
  );
};

export default GeneralSettings;
