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
  Switch,
  Col,
  TabPanel, 
  TabPanels, 
  TabGroup, 
  TabList, 
  Tab
} from "@tremor/react";
import { getCallbacksCall, setCallbacksCall, serviceHealthCheck } from "./networking";
import { Modal, Form, Input, Select, Button as Button2, message } from "antd";
import StaticGenerationSearchParamsBailoutProvider from "next/dist/client/components/static-generation-searchparams-bailout-provider";

interface SettingsPageProps {
  accessToken: string | null;
  userRole: string | null;
  userID: string | null;
}

interface AlertingVariables {
  SLACK_WEBHOOK_URL: string | null,
  LANGFUSE_PUBLIC_KEY: string | null, 
  LANGFUSE_SECRET_KEY: string | null, 
  LANGFUSE_HOST: string | null
}

interface AlertingObject {
  name: string, 
  variables: AlertingVariables
}

const Settings: React.FC<SettingsPageProps> = ({
  accessToken,
  userRole,
  userID,
}) => {
  const [callbacks, setCallbacks] = useState<any[]>([]);
  const [alerts, setAlerts] = useState<any[]>([]);
  const [isModalVisible, setIsModalVisible] = useState(false);
  const [form] = Form.useForm();
  const [selectedCallback, setSelectedCallback] = useState<string | null>(null);
  const [selectedAlertValues, setSelectedAlertValues] = useState([]);
  const [catchAllWebhookURL, setCatchAllWebhookURL] = useState<string>("");
  const [alertToWebhooks, setAlertToWebhooks] = useState<Record<string, string>>({});
  const [activeAlerts, setActiveAlerts] = useState<string[]>([]);


  const handleSwitchChange = (alertName: string) => {
    if (activeAlerts.includes(alertName)) {
      setActiveAlerts(activeAlerts.filter((alert) => alert !== alertName));
    } else {
      setActiveAlerts([...activeAlerts, alertName]);
    }
  };
  const alerts_to_UI_NAME: Record<string, string> = {
    "llm_exceptions": "LLM Exceptions",
    "llm_too_slow": "LLM Responses Too Slow",
    "llm_requests_hanging": "LLM Requests Hanging",
    "budget_alerts": "Budget Alerts (API Keys, Users)"
  }

  useEffect(() => {
    if (!accessToken || !userRole || !userID) {
      return;
    }
    getCallbacksCall(accessToken, userID, userRole).then((data) => {
      console.log("callbacks", data);
      let callbacks_data = data.callbacks;
      setCallbacks(callbacks_data);

      let alerts_data = data.alerts;
      console.log("alerts_data", alerts_data);
      if (alerts_data) {
        if (alerts_data.length > 0) {
          let _alert_info = alerts_data[0];
          console.log("_alert_info", _alert_info);
          let catch_all_webhook = _alert_info.variables.SLACK_WEBHOOK_URL;
          console.log("catch_all_webhook", catch_all_webhook);

          let active_alerts = _alert_info.active_alerts;
          setActiveAlerts(active_alerts);
          setCatchAllWebhookURL(catch_all_webhook);
          setAlertToWebhooks(_alert_info.alerts_to_webhook);

        }
      }

      setAlerts(alerts_data);
    });
  }, [accessToken, userRole, userID]);


  const isAlertOn = (alertName: string) => {
    return activeAlerts && activeAlerts.includes(alertName);
  }

  const handleAddCallback = () => {
    console.log("Add callback clicked");
    setIsModalVisible(true);
  };

  const handleCancel = () => {
    setIsModalVisible(false);
    form.resetFields();
    setSelectedCallback(null);
  };

  const handleChange = (values: any) => {
    setSelectedAlertValues(values);
    // Here, you can perform any additional logic with the selected values
    console.log('Selected values:', values);
  };

  const handleSaveAlerts = () => {
    if (!accessToken) {
      return;
    }
  
    const updatedAlertToWebhooks: Record<string, string> = {};
    Object.entries(alerts_to_UI_NAME).forEach(([key, value]) => {
      const webhookInput = document.querySelector(`input[name="${key}"]`) as HTMLInputElement;
      console.log("key", key);
      console.log("webhookInput", webhookInput);
      const newWebhookValue = webhookInput?.value || '';
      console.log("newWebhookValue", newWebhookValue);
      updatedAlertToWebhooks[key] = newWebhookValue;
    });

    console.log("updatedAlertToWebhooks", updatedAlertToWebhooks);

    const payload = {
      general_settings: {
        alert_to_webhook_url: updatedAlertToWebhooks,
        alert_types: activeAlerts
      },
    };

    console.log("payload", payload);
  
    try {
      setCallbacksCall(accessToken, payload);
    } catch (error) {
      message.error('Failed to update alerts: ' + error, 20);
    }
  
    message.success('Alerts updated successfully');
  };
  const handleSaveChanges = (callback: any) => {
    if (!accessToken) {
      return;
    }

    const updatedVariables = Object.fromEntries(
      Object.entries(callback.variables).map(([key, value]) => [key, (document.querySelector(`input[name="${key}"]`) as HTMLInputElement)?.value || value])
    );

    console.log("updatedVariables", updatedVariables);
    console.log("updateAlertTypes", selectedAlertValues);

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
        let newCallback: AlertingObject = {
          "name": values.callback,
          "variables": {
            "SLACK_WEBHOOK_URL": null,
            "LANGFUSE_HOST": null, 
            "LANGFUSE_PUBLIC_KEY": values.langfusePublicKey, 
            "LANGFUSE_SECRET_KEY": values.langfusePrivateKey
          }
        }
        // add langfuse to callbacks
        setCallbacks(callbacks ? [...callbacks, newCallback] : [newCallback]);
      } else if (values.callback === 'slack') {
        console.log(`values.slackWebhookUrl: ${values.slackWebhookUrl}`)
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
        console.log(`values.callback: ${values.callback}`)

        let newCallback: AlertingObject = {
          "name": values.callback,
          "variables": {
            "SLACK_WEBHOOK_URL": values.slackWebhookUrl,
            "LANGFUSE_HOST": null, 
            "LANGFUSE_PUBLIC_KEY": null, 
            "LANGFUSE_SECRET_KEY": null
          }
        }
        setCallbacks(callbacks ? [...callbacks, newCallback] : [newCallback]);
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

  console.log(`callbacks: ${callbacks}`)
  return (
    <div className="w-full mx-4">
      <Grid numItems={1} className="gap-2 p-8 w-full mt-2">
        <TabGroup>
        <TabList variant="line" defaultValue="1">
          <Tab value="1">Logging Callbacks</Tab>
          <Tab value="2">Alerting</Tab>
        </TabList>
        <TabPanels>
          <TabPanel>
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
          {Object.entries(callback.variables ?? {}).filter(([key, value]) => value !== null).map(([key, value]) => (
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
          </TabPanel>

          <TabPanel>

          <Card>
            <Text className="my-2">Alerts are only supported for Slack Webhook URLs. Get your webhook urls from <a href="https://api.slack.com/messaging/webhooks" target="_blank" style={{color: 'blue'}}>here</a></Text>
        <Table>
        <TableHead>
          <TableRow>
            <TableHeaderCell></TableHeaderCell>
            <TableHeaderCell></TableHeaderCell>
            <TableHeaderCell>Slack Webhook URL</TableHeaderCell>
          </TableRow>
        </TableHead>
          
        <TableBody>
          {Object.entries(alerts_to_UI_NAME).map(([key, value], index) => (
            <TableRow key={index}>
              <TableCell>
              <Switch
                    id="switch"
                    name="switch"
                    checked={isAlertOn(key)}
                    onChange={() => handleSwitchChange(key)}
                  />
              </TableCell>
              <TableCell>
              <Text>{value}</Text>
              </TableCell>
              <TableCell>
              <TextInput name={key} type="password" defaultValue={alertToWebhooks && alertToWebhooks[key] ? alertToWebhooks[key] : catchAllWebhookURL as string}>
                
              </TextInput>
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
        </Table>
        <Button size="xs" className="mt-2" onClick={handleSaveAlerts}>
          Save Changes
        </Button>

        <Button onClick={() => serviceHealthCheck(accessToken, "slack")} className="mx-2">
            Test Alerts
          </Button>

      



        </Card>
            
          </TabPanel>
          </TabPanels>
          </TabGroup>

      
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
                <TextInput type="password"/>
              </Form.Item>

              <Form.Item
                label="LANGFUSE_PRIVATE_KEY"
                name="langfusePrivateKey"
                rules={[
                  { required: true, message: "Please enter the private key" },
                ]}
              >
                <TextInput type="password"/>
              </Form.Item>
            </>
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