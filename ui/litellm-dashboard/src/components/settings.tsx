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
  Grid,
  Button,
  TextInput,
  Switch,
  TabPanel,
  TabPanels,
  TabGroup,
  TabList,
  Tab,
  SelectItem,
  Icon,
} from "@tremor/react";

import { PencilAltIcon, TrashIcon } from "@heroicons/react/outline";

import { Modal, Typography, Form, Input, Select, Button as Button2 } from "antd";
import NotificationsManager from "./molecules/notifications_manager";
import EmailSettings from "./email_settings";

const { Title, Paragraph } = Typography;

import { getCallbacksCall, setCallbacksCall, serviceHealthCheck, deleteCallback } from "./networking";
import AlertingSettings from "./alerting/alerting_settings";
import FormItem from "antd/es/form/FormItem";
import { callback_map, callbackInfo, Callbacks } from "./callback_info_helpers";
import { parseErrorMessage } from "./shared/errorUtils";
interface SettingsPageProps {
  accessToken: string | null;
  userRole: string | null;
  userID: string | null;
  premiumUser: boolean;
}

interface genericCallbackParams {
  litellm_callback_name: string; // what to send in request
  ui_callback_name: string; // what to show on UI
  litellm_callback_params: string[] | null; // known required params for this callback
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

const Settings: React.FC<SettingsPageProps> = ({ accessToken, userRole, userID, premiumUser }) => {
  const [callbacks, setCallbacks] = useState<AlertingObject[]>([]);
  const [alerts, setAlerts] = useState<any[]>([]);
  const [isModalVisible, setIsModalVisible] = useState(false);
  const [addForm] = Form.useForm();
  const [editForm] = Form.useForm();
  const [selectedCallback, setSelectedCallback] = useState<string | null>(null);
  const [catchAllWebhookURL, setCatchAllWebhookURL] = useState<string>("");
  const [alertToWebhooks, setAlertToWebhooks] = useState<Record<string, string>>({});
  const [activeAlerts, setActiveAlerts] = useState<string[]>([]);

  const [showAddCallbacksModal, setShowAddCallbacksModal] = useState(false);
  const [allCallbacks, setAllCallbacks] = useState<genericCallbackParams[]>([]);

  const [selectedCallbackParams, setSelectedCallbackParams] = useState<string[]>([]);

  const [showEditCallback, setShowEditCallback] = useState(false);
  const [selectedEditCallback, setSelectedEditCallback] = useState<any | null>(null);
  const [showDeleteConfirmModal, setShowDeleteConfirmModal] = useState(false);
  const [callbackToDelete, setCallbackToDelete] = useState<string | null>(null);

  useEffect(() => {
    if (showEditCallback && selectedEditCallback) {
      const normalized = Object.fromEntries(
        Object.entries(selectedEditCallback.variables || {}).map(([k, v]) => [k, v ?? ""]),
      );
      editForm.setFieldsValue(normalized);
    }
  }, [showEditCallback, selectedEditCallback, editForm]);

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
      setCallbacks(data.callbacks);
      setAllCallbacks(data.available_callbacks);
      // setCallbacks(callbacks_data);

      let alerts_data = data.alerts;
      if (alerts_data) {
        if (alerts_data.length > 0) {
          let _alert_info = alerts_data[0];
          let catch_all_webhook = _alert_info.variables.SLACK_WEBHOOK_URL;

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

  const updateCallbackCall = async (formValues: Record<string, any>) => {
    if (!accessToken || !selectedEditCallback) {
      return;
    }

    let env_vars: Record<string, string> = {};
    // add all other variables
    Object.entries(formValues).forEach(([key, value]) => {
      if (key !== "callback") {
        env_vars[key] = value;
      }
    });
    let payload = {
      environment_variables: formValues,
      litellm_settings: {
        success_callback: [selectedEditCallback.name],
      },
    };

    try {
      await setCallbacksCall(accessToken, payload);
      NotificationsManager.success("Callback updated successfully");
      setShowEditCallback(false);
      editForm.resetFields();
      setSelectedEditCallback(null);

      // Refresh the callbacks list
      if (userID && userRole) {
        const updatedData = await getCallbacksCall(accessToken, userID, userRole);
        setCallbacks(updatedData.callbacks);
      }
    } catch (error) {
      NotificationsManager.fromBackend(error);
    }
  };

  const addNewCallbackCall = async (formValues: Record<string, any>) => {
    if (!accessToken) {
      return;
    }
    let new_callback = formValues?.callback;

    let env_vars: Record<string, string> = {};
    // add all other variables
    Object.entries(formValues).forEach(([key, value]) => {
      if (key !== "callback") {
        env_vars[key] = value;
      }
    });

    let payload = {
      environment_variables: formValues,
      litellm_settings: {
        success_callback: [new_callback],
      },
    };

    try {
      await setCallbacksCall(accessToken, payload);
      NotificationsManager.success(`Callback ${new_callback} added successfully`);
      setShowAddCallbacksModal(false);
      addForm.resetFields();
      setSelectedCallback(null);
      setSelectedCallbackParams([]);

      // Refresh the callbacks list
      const updatedData = await getCallbacksCall(accessToken, userID || "", userRole || "");
      setCallbacks(updatedData.callbacks);
    } catch (error) {
      NotificationsManager.fromBackend(error);
    }
  };

  const handleSelectedCallbackChange = (callbackObject: genericCallbackParams) => {
    setSelectedCallback(callbackObject.litellm_callback_name);

    if (callbackObject && callbackObject.litellm_callback_params) {
      setSelectedCallbackParams(callbackObject.litellm_callback_params);
    } else {
      setSelectedCallbackParams([]);
    }
  };

  const handleSaveAlerts = async () => {
    if (!accessToken) {
      return;
    }

    const updatedAlertToWebhooks: Record<string, string> = {};
    Object.entries(alerts_to_UI_NAME).forEach(([key, value]) => {
      const webhookInput = document.querySelector(`input[name="${key}"]`) as HTMLInputElement;
      const newWebhookValue = webhookInput?.value || "";
      updatedAlertToWebhooks[key] = newWebhookValue;
    });

    const payload = {
      general_settings: {
        alert_to_webhook_url: updatedAlertToWebhooks,
        alert_types: activeAlerts,
      },
    };

    try {
      await setCallbacksCall(accessToken, payload);
    } catch (error) {
      NotificationsManager.fromBackend(error);
    }
    NotificationsManager.success("Alerts updated successfully");
  };
  const handleSaveChanges = (callback: any) => {
    if (!accessToken) {
      return;
    }

    const updatedVariables = Object.fromEntries(
      Object.entries(callback.variables).map(([key, value]) => [
        key,
        (document.querySelector(`input[name="${key}"]`) as HTMLInputElement)?.value || value,
      ]),
    );

    const payload = {
      environment_variables: updatedVariables,
      litellm_settings: {
        success_callback: [callback.name],
      },
    };

    try {
      setCallbacksCall(accessToken, payload);
    } catch (error) {
      NotificationsManager.fromBackend(error);
    }
    NotificationsManager.success("Callback updated successfully");
  };

  const handleOk = () => {
    if (!accessToken) {
      return;
    }
    // Handle form submission
    addForm.validateFields().then((values) => {
      // Call API to add the callback
      let payload;
      if (values.callback === "langfuse" || values.callback === "langfuse_otel") {
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
      addForm.resetFields();
      setSelectedCallback(null);
    });
  };

  const handleDeleteCallback = (callbackName: string) => {
    setCallbackToDelete(callbackName);
    setShowDeleteConfirmModal(true);
  };

  const confirmDeleteCallback = async () => {
    if (!callbackToDelete || !accessToken) {
      return;
    }

    try {
      await deleteCallback(accessToken, callbackToDelete);
      NotificationsManager.success(`Callback ${callbackToDelete} deleted successfully`);

      // Refresh the callbacks list
      if (userID && userRole) {
        const data = await getCallbacksCall(accessToken, userID, userRole);
        setCallbacks(data.callbacks);
      }

      setShowDeleteConfirmModal(false);
      setCallbackToDelete(null);
    } catch (error) {
      console.error("Failed to delete callback:", error);
      NotificationsManager.fromBackend(error);
    }
  };

  if (!accessToken) {
    return null;
  }

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
                      {callbacks.map((callback, index) => (
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
                              <Icon
                                icon={TrashIcon}
                                size="sm"
                                onClick={() => handleDeleteCallback(callback.name)}
                                className="text-red-500 hover:text-red-700 cursor-pointer"
                              />
                              <Button
                                onClick={async () => {
                                  try {
                                    await serviceHealthCheck(accessToken, callback.name);
                                    NotificationsManager.success("Health check triggered");
                                  } catch (error) {
                                    NotificationsManager.fromBackend(parseErrorMessage(error));
                                  }
                                }}
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
              <Button className="mt-2" onClick={() => setShowAddCallbacksModal(true)}>
                Add Callback
              </Button>
            </TabPanel>
            <TabPanel>
              <Card>
                <Text className="my-2">
                  Alerts are only supported for Slack Webhook URLs. Get your webhook urls from{" "}
                  <a href="https://api.slack.com/messaging/webhooks" target="_blank" style={{ color: "blue" }}>
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
                    {Object.entries(alerts_to_UI_NAME).map(([key, value], index) => (
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
                                <a href="https://forms.gle/W3U4PZpJGFHWtHyA9" target="_blank">
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
                    ))}
                  </TableBody>
                </Table>
                <Button size="xs" className="mt-2" onClick={handleSaveAlerts}>
                  Save Changes
                </Button>

                <Button
                  onClick={async () => {
                    try {
                      await serviceHealthCheck(accessToken, "slack");
                      NotificationsManager.success(
                        "Alert test triggered. Test request to slack made - check logs/alerts on slack to verify",
                      );
                    } catch (error) {
                      NotificationsManager.fromBackend(parseErrorMessage(error));
                    }
                  }}
                  className="mx-2"
                >
                  Test Alerts
                </Button>
              </Card>
            </TabPanel>
            <TabPanel>
              <AlertingSettings accessToken={accessToken} premiumUser={premiumUser} />
            </TabPanel>
            <TabPanel>
              <EmailSettings accessToken={accessToken} premiumUser={premiumUser} alerts={alerts} />
            </TabPanel>
          </TabPanels>
        </TabGroup>
      </Grid>

      <Modal
        title="Add Logging Callback"
        visible={showAddCallbacksModal}
        width={800}
        onCancel={() => {
          setShowAddCallbacksModal(false);
          setSelectedCallback(null);
          setSelectedCallbackParams([]);
        }}
        footer={null}
      >
        <a
          href="https://docs.litellm.ai/docs/proxy/logging"
          className="mb-8 mt-4"
          target="_blank"
          style={{ color: "blue" }}
        >
          {" "}
          LiteLLM Docs: Logging
        </a>

        <Form
          form={addForm}
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
                    handleSelectedCallbackChange(selectedCallback);
                  }
                }}
              >
                {Object.entries(Callbacks).map(([callbackEnum, callbackDisplayName]) => (
                  <SelectItem key={callbackDisplayName} value={callback_map[callbackEnum]}>
                    <div className="flex items-center space-x-2">
                      {callbackInfo[callbackDisplayName]?.logo ? (
                        <div className="w-5 h-5 flex items-center justify-center">
                          <img
                            src={callbackInfo[callbackDisplayName].logo}
                            alt={`${callbackEnum} logo`}
                            className="w-5 h-5"
                            onError={(e) => {
                              e.currentTarget.style.display = "none";
                            }}
                          />
                        </div>
                      ) : (
                        <div className="w-5 h-5 rounded-full bg-gray-200 flex items-center justify-center text-xs">
                          {(callbackDisplayName as string).charAt(0).toUpperCase()}
                        </div>
                      )}
                      <span>{callbackDisplayName}</span>
                    </div>
                  </SelectItem>
                ))}
              </Select>
            </FormItem>

            {selectedCallbackParams &&
              selectedCallbackParams.map((param) => (
                <FormItem
                  label={param}
                  name={param}
                  key={param}
                  rules={[
                    {
                      required: true,
                      message: "Please enter the value for " + param,
                    },
                  ]}
                >
                  <Input.Password />
                </FormItem>
              ))}

            <div style={{ textAlign: "right", marginTop: "10px" }}>
              <Button2 htmlType="submit">Save</Button2>
            </div>
          </>
        </Form>
      </Modal>

      <Modal
        visible={showEditCallback}
        width={800}
        title={`Edit ${selectedEditCallback?.name} Settings`}
        onCancel={() => {
          setShowEditCallback(false);
          setSelectedEditCallback(null);
        }}
        footer={null}
      >
        <Form
          form={editForm}
          onFinish={updateCallbackCall}
          labelCol={{ span: 8 }}
          wrapperCol={{ span: 16 }}
          labelAlign="left"
        >
          <>
            {selectedEditCallback &&
              selectedEditCallback.variables &&
              Object.entries(selectedEditCallback.variables).map(([param]) => (
                <FormItem
                  label={param}
                  name={param}
                  key={param}
                  rules={[
                    {
                      required: true,
                      message: `Please enter the value for ${param}`,
                    },
                  ]}
                >
                  <Input.Password />
                </FormItem>
              ))}
          </>

          <div style={{ textAlign: "right", marginTop: "10px" }}>
            <Button2 htmlType="submit">Save</Button2>
          </div>
        </Form>
      </Modal>

      <Modal
        title="Confirm Delete"
        visible={showDeleteConfirmModal}
        onOk={confirmDeleteCallback}
        onCancel={() => {
          setShowDeleteConfirmModal(false);
          setCallbackToDelete(null);
        }}
        okText="Delete"
        cancelText="Cancel"
        okButtonProps={{ danger: true }}
      >
        <p>Are you sure you want to delete the callback - {callbackToDelete}? This action cannot be undone.</p>
      </Modal>
    </div>
  );
};

export default Settings;
