import React, { useState, useEffect } from "react";
import {
  Card,

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
  Tab,
  Callout,
  SelectItem,
  Icon,
} from "@tremor/react";

import {
  PencilAltIcon
} from "@heroicons/react/outline";

import { Modal, Typography, Form, Input, Select, Button as Button2, message } from "antd";

const { Title, Paragraph } = Typography;
const isLocal = process.env.NODE_ENV === "development";
const proxyBaseUrl = isLocal ? "http://localhost:4000" : null;
if (isLocal != true) {
  console.log = function() {};
}
import {
  getCallbacksCall,
  setCallbacksCall,
  serviceHealthCheck,
} from "./networking";
import StaticGenerationSearchParamsBailoutProvider from "next/dist/client/components/static-generation-searchparams-bailout-provider";
import AlertingSettings from "./alerting/alerting_settings";
import FormItem from "antd/es/form/FormItem";
interface SettingsPageProps {
  accessToken: string | null;
  userRole: string | null;
  userID: string | null;
  premiumUser: boolean;
}


interface genericCallbackParams {
  
  litellm_callback_name: string  // what to send in request
  ui_callback_name: string // what to show on UI
  litellm_callback_params: string[] | null // known required params for this callback
}


interface AlertingVariables {
  SLACK_WEBHOOK_URL: string | null;
  LANGFUSE_PUBLIC_KEY: string | null;
  LANGFUSE_SECRET_KEY: string | null;
  LANGFUSE_HOST: string | null;
  OPENMETER_API_KEY: string | null;
}

interface AlertingObject {
  name: string;
  variables: AlertingVariables;
}

const defaultLoggingObject: AlertingObject[] = [
  {
    name: "slack",
    variables: {
      LANGFUSE_HOST: null,
      LANGFUSE_PUBLIC_KEY: null,
      LANGFUSE_SECRET_KEY: null,
      OPENMETER_API_KEY: null,
      SLACK_WEBHOOK_URL: null,
    },
  },
  {
    name: "langfuse",
    variables: {
      LANGFUSE_HOST: null,
      LANGFUSE_PUBLIC_KEY: null,
      LANGFUSE_SECRET_KEY: null,
      OPENMETER_API_KEY: null,
      SLACK_WEBHOOK_URL: null,
    },
  },
  {
    name: "openmeter",
    variables: {
      LANGFUSE_HOST: null,
      LANGFUSE_PUBLIC_KEY: null,
      LANGFUSE_SECRET_KEY: null,
      OPENMETER_API_KEY: null,
      SLACK_WEBHOOK_URL: null,
    },
  },
];

const Settings: React.FC<SettingsPageProps> = ({
  accessToken,
  userRole,
  userID,
  premiumUser,
}) => {
  const [callbacks, setCallbacks] =
    useState<AlertingObject[]>([]);
  const [alerts, setAlerts] = useState<any[]>([]);
  const [isModalVisible, setIsModalVisible] = useState(false);
  const [form] = Form.useForm();
  const [selectedCallback, setSelectedCallback] = useState<string | null>(null);
  const [selectedAlertValues, setSelectedAlertValues] = useState([]);
  const [catchAllWebhookURL, setCatchAllWebhookURL] = useState<string>("");
  const [alertToWebhooks, setAlertToWebhooks] = useState<
    Record<string, string>
  >({});
  const [activeAlerts, setActiveAlerts] = useState<string[]>([]);

  const [showAddCallbacksModal, setShowAddCallbacksModal] = useState(false);
  const [allCallbacks, setAllCallbacks] = useState<genericCallbackParams[]>([]);

  const [selectedCallbacktoAdd, setSelectedCallbacktoAdd] = useState<string | null>(null);
  const [selectedCallbackParams, setSelectedCallbackParams] = useState<string[]>([]);

  const [showEditCallback, setShowEditCallback] = useState(false);
  const [selectedEditCallback, setSelectedEditCallback] = useState<any | null>(null);

  const handleSwitchChange = (alertName: string) => {
    if (activeAlerts.includes(alertName)) {
      setActiveAlerts(activeAlerts.filter((alert) => alert !== alertName));
    } else {
      setActiveAlerts([...activeAlerts, alertName]);
    }
  };
  const alerts_to_UI_NAME: Record<string, string> = {
    llm_exceptions: "LLM Exceptions",
    llm_too_slow: "LLM Responses Too Slow",
    llm_requests_hanging: "LLM Requests Hanging",
    budget_alerts: "Budget Alerts (API Keys, Users)",
    db_exceptions: "Database Exceptions (Read/Write)",
    daily_reports: "Weekly/Monthly Spend Reports",
    outage_alerts: "Outage Alerts",
    region_outage_alerts: "Region Outage Alerts",
  };

  useEffect(() => {
    if (!accessToken || !userRole || !userID) {
      return;
    }
    getCallbacksCall(accessToken, userID, userRole).then((data) => {
      console.log("callbacks", data);

      setCallbacks(data.callbacks);
      setAllCallbacks(data.available_callbacks);
      // setCallbacks(callbacks_data);

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
  };

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
    console.log("Selected values:", values);
  };

  const handleSaveEmailSettings = () => {
    if (!accessToken) {
      return;
    }


    let updatedVariables: Record<string, string> = {};

    alerts
      .filter((alert) => alert.name === "email")
      .forEach((alert) => {
        Object.entries(alert.variables ?? {}).forEach(([key, value]) => {
          const inputElement = document.querySelector(`input[name="${key}"]`) as HTMLInputElement;
          if (inputElement && inputElement.value) {
            updatedVariables[key] = inputElement?.value;
          }
        });
      });

    console.log("updatedVariables", updatedVariables);
    //filter out null / undefined values for updatedVariables

    const payload = {
      general_settings: {
        alerting: ["email"],
      },
      environment_variables: updatedVariables,
    };
    try {
      setCallbacksCall(accessToken, payload);
    } catch (error) {
      message.error("Failed to update alerts: " + error, 20);
    }

    message.success("Email settings updated successfully");
  }

  const updateCallbackCall = async (formValues: Record<string, any>) => {
    if (!accessToken) {
      return;
    }

    let env_vars: Record<string, string> = {};
    // add all other variables
    Object.entries(formValues).forEach(([key, value]) => {
      if (key !== "callback") {
        env_vars[key] = value
      }
    });
    let payload = {
      environment_variables: env_vars,
    }


    try {
      let newCallback = await setCallbacksCall(accessToken, payload);
      message.success(`Callback added successfully`);
      setIsModalVisible(false);
      form.resetFields();
      setSelectedCallback(null);
    } catch (error) {
      message.error("Failed to add callback: " + error, 20);
    }
  }




  const addNewCallbackCall = async (formValues: Record<string, any>) => {
    if (!accessToken) {
      return;
    }
    let new_callback = formValues?.callback

    let env_vars: Record<string, string> = {};
    // add all other variables
    Object.entries(formValues).forEach(([key, value]) => {
      if (key !== "callback") {
        env_vars[key] = value
      }
    });

    let payload = {
      environment_variables: env_vars,
      litellm_settings: {
        success_callback: [new_callback],
      },
    }


    try {
      let newCallback = await setCallbacksCall(accessToken, payload);
      message.success(`Callback ${new_callback} added successfully`);
      setIsModalVisible(false);
      form.resetFields();
      setSelectedCallback(null);
    } catch (error) {
      message.error("Failed to add callback: " + error, 20);
    }
  }



  const handleSelectedCallbackChange = (callbackObject: genericCallbackParams) => {

    console.log("inside handleSelectedCallbackChange", callbackObject);
    setSelectedCallback(callbackObject.litellm_callback_name);

    console.log("all callbacks", allCallbacks);
    if (callbackObject && callbackObject.litellm_callback_params) {
      setSelectedCallbackParams(callbackObject.litellm_callback_params);
      console.log("selectedCallbackParams", selectedCallbackParams);
    } else {
      setSelectedCallbackParams([]);
    }
  };

  const handleSaveAlerts = () => {
    if (!accessToken) {
      return;
    }

    const updatedAlertToWebhooks: Record<string, string> = {};
    Object.entries(alerts_to_UI_NAME).forEach(([key, value]) => {
      const webhookInput = document.querySelector(
        `input[name="${key}"]`
      ) as HTMLInputElement;
      console.log("key", key);
      console.log("webhookInput", webhookInput);
      const newWebhookValue = webhookInput?.value || "";
      console.log("newWebhookValue", newWebhookValue);
      updatedAlertToWebhooks[key] = newWebhookValue;
    });

    console.log("updatedAlertToWebhooks", updatedAlertToWebhooks);

    const payload = {
      general_settings: {
        alert_to_webhook_url: updatedAlertToWebhooks,
        alert_types: activeAlerts,
      },
    };

    console.log("payload", payload);

    try {
      setCallbacksCall(accessToken, payload);
    } catch (error) {
      message.error("Failed to update alerts: " + error, 20);
    }

    message.success("Alerts updated successfully");
  };
  const handleSaveChanges = (callback: any) => {
    if (!accessToken) {
      return;
    }

    const updatedVariables = Object.fromEntries(
      Object.entries(callback.variables).map(([key, value]) => [
        key,
        (document.querySelector(`input[name="${key}"]`) as HTMLInputElement)
          ?.value || value,
      ])
    );

    console.log("updatedVariables", updatedVariables);
    console.log("updateAlertTypes", selectedAlertValues);

    const payload = {
      environment_variables: updatedVariables,
      litellm_settings: {
        success_callback: [callback.name],
      },
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
      if (values.callback === "langfuse") {
        payload = {
          environment_variables: {
            LANGFUSE_PUBLIC_KEY: values.langfusePublicKey,
            LANGFUSE_SECRET_KEY: values.langfusePrivateKey,
          },
          litellm_settings: {
            success_callback: [values.callback],
          },
        };
        setCallbacksCall(accessToken, payload);
        let newCallback: AlertingObject = {
          name: values.callback,
          variables: {
            SLACK_WEBHOOK_URL: null,
            LANGFUSE_HOST: null,
            LANGFUSE_PUBLIC_KEY: values.langfusePublicKey,
            LANGFUSE_SECRET_KEY: values.langfusePrivateKey,
            OPENMETER_API_KEY: null,
          },
        };
        // add langfuse to callbacks
        setCallbacks(callbacks ? [...callbacks, newCallback] : [newCallback]);
      } else if (values.callback === "slack") {
        console.log(`values.slackWebhookUrl: ${values.slackWebhookUrl}`);
        payload = {
          general_settings: {
            alerting: ["slack"],
            alerting_threshold: 300,
          },
          environment_variables: {
            SLACK_WEBHOOK_URL: values.slackWebhookUrl,
          },
        };
        setCallbacksCall(accessToken, payload);

        // add slack to callbacks
        console.log(`values.callback: ${values.callback}`);

        let newCallback: AlertingObject = {
          name: values.callback,
          variables: {
            SLACK_WEBHOOK_URL: values.slackWebhookUrl,
            LANGFUSE_HOST: null,
            LANGFUSE_PUBLIC_KEY: null,
            LANGFUSE_SECRET_KEY: null,
            OPENMETER_API_KEY: null,
          },
        };
        setCallbacks(callbacks ? [...callbacks, newCallback] : [newCallback]);
      } else if (values.callback == "openmeter") {
        console.log(`values.openMeterApiKey: ${values.openMeterApiKey}`);
        payload = {
          environment_variables: {
            OPENMETER_API_KEY: values.openMeterApiKey,
          },
          litellm_settings: {
            success_callback: [values.callback],
          },
        };
        setCallbacksCall(accessToken, payload);
        let newCallback: AlertingObject = {
          name: values.callback,
          variables: {
            SLACK_WEBHOOK_URL: null,
            LANGFUSE_HOST: null,
            LANGFUSE_PUBLIC_KEY: null,
            LANGFUSE_SECRET_KEY: null,
            OPENMETER_API_KEY: values.openMeterAPIKey,
          },
        };
        // add langfuse to callbacks
        setCallbacks(callbacks ? [...callbacks, newCallback] : [newCallback]);
      } else {
        payload = {
          error: "Invalid callback value",
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

  console.log(`callbacks: ${callbacks}`);
  return (
    <div className="w-full mx-4">
      <Grid numItems={1} className="gap-2 p-8 w-full mt-2">
        
        <TabGroup>
          <TabList variant="line" defaultValue="1">
            <Tab value="1">Logging Callbacks</Tab>
            <Tab value="2">Alerting Types</Tab>
            <Tab value="3">Alerting Settings</Tab>
            <Tab value="4">Email Alerts</Tab>
          </TabList>
          <TabPanels>
            <TabPanel>
            <Title level={4}>Active Logging Callbacks</Title>

            <Grid numItems={2}>

              <Card className="max-h-[50vh]">
                
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableHeaderCell>Callback Name</TableHeaderCell>
                      {/* <TableHeaderCell>Callback Env Vars</TableHeaderCell> */}

                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {callbacks
                      .map((callback, index) => (
                        <TableRow key={index} className="flex justify-between">

                          <TableCell>
                            <Text>{callback.name}</Text>
                          </TableCell>
                          <TableCell>
                            <Grid numItems={2} className="flex justify-between">
                            <Icon
                                icon={PencilAltIcon}
                                size="sm"
                                onClick={() => {
                                  setSelectedEditCallback(callback);
                                  setShowEditCallback(true);
                                }}
                              />
                              <Button
                                onClick={() =>
                                  serviceHealthCheck(accessToken, callback.name)
                                }
                                className="ml-2"
                                variant="secondary"
                              >
                                Test Callback
                              </Button>
                            </Grid>
                          </TableCell>
                        </TableRow>
                      ))}
                  </TableBody>
                </Table>
              </Card>
              </Grid>
              <Button
              className="mt-2"
                onClick={() => setShowAddCallbacksModal(true)}>
                  Add Callback
                </Button>
            </TabPanel>
            <TabPanel>
              <Card>
                <Text className="my-2">
                  Alerts are only supported for Slack Webhook URLs. Get your
                  webhook urls from{" "}
                  <a
                    href="https://api.slack.com/messaging/webhooks"
                    target="_blank"
                    style={{ color: "blue" }}
                  >
                    here
                  </a>
                </Text>
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableHeaderCell></TableHeaderCell>
                      <TableHeaderCell></TableHeaderCell>
                      <TableHeaderCell>Slack Webhook URL</TableHeaderCell>
                    </TableRow>
                  </TableHead>

                  <TableBody>
                    {Object.entries(alerts_to_UI_NAME).map(
                      ([key, value], index) => (
                        <TableRow key={index}>
                          <TableCell>
                            {key == "region_outage_alerts" ? (
                              premiumUser ? (
                                <Switch
                                  id="switch"
                                  name="switch"
                                  checked={isAlertOn(key)}
                                  onChange={() => handleSwitchChange(key)}
                                />
                              ) : (
                                <Button className="flex items-center justify-center">
                                  <a
                                    href="https://forms.gle/W3U4PZpJGFHWtHyA9"
                                    target="_blank"
                                  >
                                    ✨ Enterprise Feature
                                  </a>
                                </Button>
                              )
                            ) : (
                              <Switch
                                id="switch"
                                name="switch"
                                checked={isAlertOn(key)}
                                onChange={() => handleSwitchChange(key)}
                              />
                            )}
                          </TableCell>
                          <TableCell>
                            <Text>{value}</Text>
                          </TableCell>
                          <TableCell>
                            <TextInput
                              name={key}
                              type="password"
                              defaultValue={
                                alertToWebhooks && alertToWebhooks[key]
                                  ? alertToWebhooks[key]
                                  : (catchAllWebhookURL as string)
                              }
                            ></TextInput>
                          </TableCell>
                        </TableRow>
                      )
                    )}
                  </TableBody>
                </Table>
                <Button size="xs" className="mt-2" onClick={handleSaveAlerts}>
                  Save Changes
                </Button>

                <Button
                  onClick={() => serviceHealthCheck(accessToken, "slack")}
                  className="mx-2"
                >
                  Test Alerts
                </Button>
              </Card>
            </TabPanel>
            <TabPanel>
              <AlertingSettings
                accessToken={accessToken}
                premiumUser={premiumUser}
              />
            </TabPanel>
            <TabPanel>
              <Card>
      <Title level={4}>Email Settings</Title>
      <Text>
      <a href="https://docs.litellm.ai/docs/proxy/email" target="_blank" style={{ color: "blue" }}> LiteLLM Docs: email alerts</a> <br/>        
      </Text>
<div className="flex w-full">
  {alerts
    .filter((alert) => alert.name === "email")
    .map((alert, index) => (
      <TableCell key={index}>

        <ul>
        <Grid numItems={2}>
          {Object.entries(alert.variables ?? {}).map(([key, value]) => (
            <li key={key} className="mx-2 my-2">
              
              { premiumUser!= true && (key === "EMAIL_LOGO_URL" || key === "EMAIL_SUPPORT_CONTACT") ? (
                <div>
                  <a
                  href="https://forms.gle/W3U4PZpJGFHWtHyA9"
                  target="_blank"
                >
                  <Text className="mt-2">
                  {" "}
                  ✨ {key}

                  </Text>
                 
                </a>
                <TextInput
                name={key}
                defaultValue={value as string}
                type="password"
                disabled={true}
                style={{ width: "400px" }}
              />
                </div>
                
              ) : (
                <div>
                  <Text className="mt-2">{key}</Text>
                  <TextInput
                name={key}
                defaultValue={value as string}
                type="password"
                style={{ width: "400px" }}
              />
                </div>
                
              )}
              
              {/* Added descriptions for input fields */}
              <p style={{ fontSize: "small", fontStyle: "italic" }}>
                {key === "SMTP_HOST" && (
                  <div style={{ color: "gray" }}>
                    Enter the SMTP host address, e.g. `smtp.resend.com`
                    <span style={{ color: "red" }}> Required * </span>
                  </div>
                  
                )}

                {key === "SMTP_PORT" && (
                  <div style={{ color: "gray" }}>
                    Enter the SMTP port number, e.g. `587`
                     <span style={{ color: "red" }}> Required * </span>

                  </div>
                 
                )}
                
                {key === "SMTP_USERNAME" && (
                  <div style={{ color: "gray" }}>
                    Enter the SMTP username, e.g. `username`
                    <span style={{ color: "red" }}> Required * </span>
                  </div>
                  
                )}
                
                {key === "SMTP_PASSWORD" && (
                  <span style={{ color: "red" }}> Required * </span>
                )}

                {key === "SMTP_SENDER_EMAIL" && (
                  <div style={{ color: "gray" }}>
                    Enter the sender email address, e.g. `sender@berri.ai`
                  <span style={{ color: "red" }}> Required * </span>

                  </div>
                )}

                {key === "TEST_EMAIL_ADDRESS" && (
                  <div style={{ color: "gray" }}>
                  Email Address to send `Test Email Alert` to. example: `info@berri.ai`
                  <span style={{ color: "red" }}> Required * </span>
                  </div>
                )
                }
                {key === "EMAIL_LOGO_URL" && (
                  <div style={{ color: "gray" }}>
                   (Optional) Customize the Logo that appears in the email, pass a url to your logo
                  </div>
                )
                }
                {key === "EMAIL_SUPPORT_CONTACT" && (
                  <div style={{ color: "gray" }}>
                   (Optional) Customize the support email address that appears in the email. Default is support@berri.ai
                  </div>
                )
                   }
              </p>
            </li>
          ))}
                  </Grid>
        </ul>
      </TableCell>
    ))}
</div>

            <Button
              className="mt-2"
              onClick={() => handleSaveEmailSettings()}
            >
              Save Changes
            </Button>
            <Button
              onClick={() =>
                serviceHealthCheck(accessToken, "email")
              }
              className="mx-2"
            >
              Test Email Alerts
            </Button>
            
              </Card>
          </TabPanel>
          </TabPanels>
        </TabGroup>
      </Grid>

      
      <Modal
      title="Add Logging Callback"
      visible={showAddCallbacksModal}
      width={800}
      onCancel= {() => setShowAddCallbacksModal(false)}
      footer={null}
      >
        
      <a href="https://docs.litellm.ai/docs/proxy/logging" className="mb-8 mt-4" target="_blank" style={{ color: "blue" }}> LiteLLM Docs: Logging</a>


      <Form 
      form={form} 
      onFinish={addNewCallbackCall}
      labelCol={{ span: 8 }}
      wrapperCol={{ span: 16 }}
      labelAlign="left"
      >

      <>
        <FormItem
          label="Callback"
          name="callback"
          rules={[{ required: true, message: "Please select a callback" }]}
        >
          <Select
          onChange={(value) => {
            const selectedCallback = allCallbacks[value];
            if (selectedCallback) {
              console.log(selectedCallback.ui_callback_name);
              handleSelectedCallbackChange(selectedCallback);
            }
          }}
          >
        {allCallbacks &&
          Object.values(allCallbacks).map((callback) => (
            <SelectItem 
            key={callback.litellm_callback_name} 
            value={callback.litellm_callback_name}
            >
              {callback.ui_callback_name}
            </SelectItem>
          ))}
      </Select>
        </FormItem>


        {
          selectedCallbackParams && selectedCallbackParams.map((param) => (
            <FormItem
              label={param}
              name={param}
              key={param}
              rules={[{ required: true, message: "Please enter the value for " + param}]}
            >
              <TextInput type="password" />
            </FormItem>
          ))
        }

          <div style={{ textAlign: "right", marginTop: "10px" }}>
              <Button2 htmlType="submit">Save</Button2>
          </div>

          </>
        </Form>
    </Modal>

    <Modal
    visible={showEditCallback}
    width={800}
    title={`Edit ${selectedEditCallback?.name } Settings`}
    onCancel= {() => setShowEditCallback(false)}
    footer={null}
    >

      <Form 
      form={form} 
      onFinish={updateCallbackCall}
      labelCol={{ span: 8 }}
      wrapperCol={{ span: 16 }}
      labelAlign="left"
      >
      <>
      {
        selectedEditCallback && selectedEditCallback.variables && Object.entries(selectedEditCallback.variables).map(([param, value]) => (
          <FormItem
            label={param}
            name={param}
            key={param}
          >
            <TextInput type="password" defaultValue={value as string}/>
          </FormItem>
        ))
      }

      </>

      <div style={{ textAlign: "right", marginTop: "10px" }}>
              <Button2 htmlType="submit">Save</Button2>
          </div>

      </Form>
        

    </Modal>
    </div>
    
  );
};

export default Settings;
