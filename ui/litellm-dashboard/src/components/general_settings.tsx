import React, { useState, useEffect } from "react";
import {
  Card,
  Title,
  Subtitle,
  Table,
  TableHead,
  TableRow,
  Badge,
  TableHeaderCell,
  TableCell,
  TableBody,
  Metric,
  Text,
  Grid,
  Button,
  TextInput,
  Col,
} from "@tremor/react";
import { getCallbacksCall, setCallbacksCall, serviceHealthCheck } from "./networking";
import { Modal, Form, Input, Select, Button as Button2, message } from "antd";
import StaticGenerationSearchParamsBailoutProvider from "next/dist/client/components/static-generation-searchparams-bailout-provider";

interface GeneralSettingsPageProps {
  accessToken: string | null;
  userRole: string | null;
  userID: string | null;
}

const GeneralSettings: React.FC<GeneralSettingsPageProps> = ({
  accessToken,
  userRole,
  userID,
}) => {
  const [routerSettings, setRouterSettings] = useState<{ [key: string]: any }>({});
  const [isModalVisible, setIsModalVisible] = useState(false);
  const [form] = Form.useForm();
  const [selectedCallback, setSelectedCallback] = useState<string | null>(null);

  useEffect(() => {
    if (!accessToken || !userRole || !userID) {
      return;
    }
    getCallbacksCall(accessToken, userID, userRole).then((data) => {
      console.log("callbacks", data);
      let router_settings = data.router_settings;
      setRouterSettings(router_settings);
    });
  }, [accessToken, userRole, userID]);

  const handleAddCallback = () => {
    console.log("Add callback clicked");
    setIsModalVisible(true);
  };

  const handleCancel = () => {
    setIsModalVisible(false);
    form.resetFields();
    setSelectedCallback(null);
  };

  const handleSaveChanges = (router_settings: any) => {
    if (!accessToken) {
      return;
    }

    console.log("router_settings", router_settings);

    const updatedVariables = Object.fromEntries(
      Object.entries(router_settings).map(([key, value]) => [key, (document.querySelector(`input[name="${key}"]`) as HTMLInputElement)?.value || value])
    );

    console.log("updatedVariables", updatedVariables);

    const payload = {
      router_settings: updatedVariables
    };

    try {
      setCallbacksCall(accessToken, payload);
    } catch (error) {
      message.error("Failed to update router settings: " + error, 20);
    }

    message.success("router settings updated successfully");
  };

  

  if (!accessToken) {
    return null;
  }

  return (
    <div className="w-full mx-4">
      <Grid numItems={1} className="gap-2 p-8 w-full mt-2">
      <Title>Router Settings</Title>
        <Card >
          <Table>
            <TableHead>
              <TableRow>
                <TableHeaderCell>Setting</TableHeaderCell>
                <TableHeaderCell>Value</TableHeaderCell>
              </TableRow>
            </TableHead>
            <TableBody>
            {Object.entries(routerSettings).map(([param, value]) => (
  <TableRow key={param}>
    <TableCell>
      <Text>{param}</Text>
    </TableCell>
    <TableCell>
      <TextInput
        name={param}
        defaultValue={
          typeof value === 'object' ? JSON.stringify(value, null, 2) : value.toString()
        }
      />
    </TableCell>
  </TableRow>
))}
</TableBody>
        </Table>
        </Card>
        <Col>
            <Button className="mt-2" onClick={() => handleSaveChanges(routerSettings)}>
            Save Changes
            </Button>
        </Col>
      </Grid>

    </div>
  );
};

export default GeneralSettings;