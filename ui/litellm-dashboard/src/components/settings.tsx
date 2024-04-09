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
  Col,
} from "@tremor/react";
import { getCallbacksCall, setCallbacksCall } from "./networking";
import { Modal, Form, Input, Select, Button as Button2 } from "antd";
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
  const [callbacks, setCallbacks] = useState<string[]>([]);
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
      let callback_names = callbacks_data.success_callback; // ["callback1", "callback2"]
      setCallbacks(callback_names);
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
          litellm_settings: {
            success_callback: [values.callback]
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

  return (
    <div className="w-full mx-4">
      <Grid numItems={1} className="gap-2 p-8 h-[75vh] w-full mt-2">
        <Card className="h-[15vh]">
          <Grid numItems={2} className="mt-2">
            <Col>
              <Title>Logging Callbacks</Title>
            </Col>
            <Col>
            <div>
            {!callbacks ? (
                <Badge color={"red"}>None</Badge>
                ) : callbacks.length === 0 ? (
                <Badge>None</Badge>
                ) : (
                callbacks.map((callback, index) => (
                    <Badge key={index} color={"sky"}>
                    {callback}
                    </Badge>
                ))
                )}
            </div>
            </Col>
          </Grid>
          <Col>
            <Button size="xs" className="mt-2" onClick={handleAddCallback}>
              Add Callback
            </Button>
          </Col>
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