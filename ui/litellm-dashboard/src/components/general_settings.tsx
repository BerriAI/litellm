import React, { useState, useEffect } from "react";
import {
  Card,
  Table,
  TableHead,
  TableRow,
  Badge,
  TableHeaderCell,
  TableCell,
  TableBody,
  Text,
  Button,
  Icon,
} from "@tremor/react";
import { TabPanel, TabPanels, TabGroup, TabList, Tab } from "@tremor/react";
import {
  getGeneralSettingsCall,
  updateConfigFieldSetting,
  deleteConfigFieldSetting,
} from "./networking";
import { InputNumber } from "antd";
import { TrashIcon, CheckCircleIcon } from "@heroicons/react/outline";

import RouterSettings from "./router_settings";
import Fallbacks from "./fallbacks";
interface GeneralSettingsPageProps {
  accessToken: string | null;
  userRole: string | null;
  userID: string | null;
  modelData: any;
}

interface generalSettingsItem {
  field_name: string;
  field_type: string;
  field_value: any;
  field_description: string;
  stored_in_db: boolean | null;
}

const GeneralSettings: React.FC<GeneralSettingsPageProps> = ({ accessToken, userRole, userID, modelData }) => {
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
          <Tab value="2">Fallbacks</Tab>
          <Tab value="3">General</Tab>
        </TabList>
        <TabPanels className="px-8 py-6">
          <TabPanel>
            <RouterSettings
              accessToken={accessToken}
              userRole={userRole}
              userID={userID}
              modelData={modelData}
            />
          </TabPanel>
          <TabPanel>
            <Fallbacks
              accessToken={accessToken}
              userRole={userRole}
              userID={userID}
              modelData={modelData}
            />
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
                          {value.field_type == "Integer" ? (
                            <InputNumber
                              step={1}
                              value={value.field_value}
                              onChange={(newValue) => handleInputChange(value.field_name, newValue)} // Handle value change
                            />
                          ) : null}
                        </TableCell>
                        <TableCell>
                          {value.stored_in_db == true ? (
                            <Badge icon={CheckCircleIcon} className="text-white">
                              In DB
                            </Badge>
                          ) : value.stored_in_db == false ? (
                            <Badge className="text-gray bg-white outline">In Config</Badge>
                          ) : (
                            <Badge className="text-gray bg-white outline">Not Set</Badge>
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
