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

interface SettingsPageProps {
  accessToken: string | null;
  userRole: string | null;
  userID: string | null;
}

const Settings: React.FC<SettingsPageProps> = ({
  accessToken,
  userRole,
  userID,
}) => {
  const [callbacks, setCallbacks] = useState<any[]>([]);
  const [isModalVisible, setIsModalVisible] = useState(false);
  const [form] = Form.useForm();
  const [selectedCallback, setSelectedCallback] = useState<string | null>(null);

  useEffect(() => {
    if (!accessToken || !userRole || !userID) {
      return;
    }
    getCallbacksCall(accessToken, userID, userRole).then((data) => {
      console.log("callbacks", data);
      let callbacks_data = data.data;
      setCallbacks(callbacks_data);
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

  const handleSaveChanges = (callback: any) => {
    if (!accessToken) {
      return;
    }

    const updatedVariables = Object.fromEntries(
      Object.entries(callback.variables).map(([key, value]) => [key, (document.querySelector(`input[name="${key}"]`) as HTMLInputElement)?.value || value])
    );

    const payload = {
      environment_variables: updatedVariables,
    };

    try {
      setCallbacksCall(accessToken, payload);
    } catch (error) {
      message.error("Failed to update callback: " + error, 20);
    }

    message.success("Callback updated successfully");
  };

  const handleOk = () => {
    if (!accessToken) {
      return;
    }
    // Handle form submission
    form.validateFields().then((values) => {
      // Call API to add the callback
      console.log("Form values:", values);
      let payload;
      if (values.callback === 'langfuse') {
        payload = {
          environment_variables: {
            LANGFUSE_PUBLIC_KEY: values.langfusePublicKey,
            LANGFUSE_SECRET_KEY: values.langfusePrivateKey
          },
          litellm_settings: {
            success_callback: [values.callback]
          }
        };
        setCallbacksCall(accessToken, payload);

        // add langfuse to callbacks
        setCallbacks(callbacks ? [...callbacks, values.callback] : [values.callback]);
      } else if (values.callback === 'slack') {
        payload = {
          general_settings: {
            alerting: ["slack"],
            alerting_threshold: 300
          },
          environment_variables: {
            SLACK_WEBHOOK_URL: values.slackWebhookUrl
          }
        };
        setCallbacksCall(accessToken, payload);

        // add slack to callbacks
        setCallbacks(callbacks ? [...callbacks, values.callback] : [values.callback]);
      } else {
        payload = {
          error: 'Invalid callback value'
        };
      }
      setIsModalVisible(false);
      form.resetFields();
      setSelectedCallback(null);
    });
  };

  const handleCallbackChange = (value: string) => {
    setSelectedCallback(value);
  };

  if (!accessToken) {
    return null;
  }

  return (
    <div className="w-full mx-4">
      <Grid numItems={1} className="gap-2 p-8 w-full mt-2">
      <Title>Logging Callbacks</Title>
        <Card >
          <Table>
            <TableHead>
              <TableRow>
                <TableHeaderCell>Callback</TableHeaderCell>
                <TableHeaderCell>Callback Env Vars</TableHeaderCell>
              </TableRow>
            </TableHead>
            <TableBody>
  {callbacks.map((callback, index) => (
    <TableRow key={index}>
      <TableCell>
        <Badge color="emerald">{callback.name}</Badge>
      </TableCell>
      <TableCell>
        <ul>
        {Object.entries(callback.variables).map(([key, value]) => (
  <li key={key}>
    <Text className="mt-2">{key}</Text>
    {key === "LANGFUSE_HOST" ? (
      <p>default value=https://cloud.langfuse.com</p>
    ) : (
      <div></div>
    )}
    <TextInput name={key} defaultValue={value as string} type="password" />
  </li>
))}
        </ul>
        <Button className="mt-2" onClick={() => handleSaveChanges(callback)}>
          Save Changes
        </Button>
        <Button onClick={() => serviceHealthCheck(accessToken, callback.name)} className="mx-2">
          Test Callback
        </Button>
      </TableCell>
    </TableRow>
  ))}
</TableBody>
          </Table>
            <Button size="xs" className="mt-2" onClick={handleAddCallback}>
              Add Callback
            </Button>
            
        </Card>
        
      </Grid>

      <Modal
        title="Add Callback"
        visible={isModalVisible}
        onOk={handleOk}
        width={800}
        onCancel={handleCancel}
        footer={null}
      >
        <Form form={form} layout="vertical" onFinish={handleOk}>
          <Form.Item
            label="Callback"
            name="callback"
            rules={[{ required: true, message: "Please select a callback" }]}
          >
            <Select onChange={handleCallbackChange}>
              <Select.Option value="langfuse">langfuse</Select.Option>
              <Select.Option value="slack">slack alerting</Select.Option>
            </Select>
          </Form.Item>

          {selectedCallback === 'langfuse' && (
            <>
              <Form.Item
                label="LANGFUSE_PUBLIC_KEY"
                name="langfusePublicKey"
                rules={[
                  { required: true, message: "Please enter the public key" },
                ]}
              >
                <Input.Password />
              </Form.Item>

              <Form.Item
                label="LANGFUSE_PRIVATE_KEY"
                name="langfusePrivateKey"
                rules={[
                  { required: true, message: "Please enter the private key" },
                ]}
              >
                <Input.Password />
              </Form.Item>
            </>
          )}

          {selectedCallback === 'slack' && (
            <Form.Item
              label="SLACK_WEBHOOK_URL"
              name="slackWebhookUrl"
              rules={[
                { required: true, message: "Please enter the Slack webhook URL" },
              ]}
            >
              <Input />
            </Form.Item>
          )}

          <div style={{ textAlign: "right", marginTop: "10px" }}>
            <Button2 htmlType="submit">Save</Button2>
          </div>
        </Form>
      </Modal>
    </div>
  );
};

export default Settings;