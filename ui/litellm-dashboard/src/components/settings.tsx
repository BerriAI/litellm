import {
  Button,
  Card,
  Grid,
  SelectItem,
  Switch,
  Tab,
  TabGroup,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeaderCell,
  TableRow,
  TabList,
  TabPanel,
  TabPanels,
  Text,
  TextInput,
} from "@tremor/react";
import React, { useEffect, useState } from "react";

import { Button as Button2, Form, Input, Modal, Select, Typography } from "antd";
import EmailSettings from "./email_settings";
import NotificationsManager from "./molecules/notifications_manager";

const { Title, Paragraph } = Typography;

import FormItem from "antd/es/form/FormItem";
import AlertingSettings from "./alerting/alerting_settings";
import {
  deleteCallback,
  getCallbackConfigsCall,
  getCallbacksCall,
  serviceHealthCheck,
  setCallbacksCall,
} from "./networking";
import { LoggingCallbacksTable } from "./Settings/LoggingAndAlerts/LoggingCallbacks/LoggingCallbacksTable";
import { AlertingObject } from "./Settings/LoggingAndAlerts/LoggingCallbacks/types";
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

const assetsLogoFolder = "../ui/assets/logos/";

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
  const [callbackConfigs, setCallbackConfigs] = useState<any[]>([]);
  const [allCallbacks, setAllCallbacks] = useState<
    Record<
      string,
      {
        litellm_callback_name: string;
        litellm_callback_params: string[];
        ui_callback_name: string;
      }
    >
  >({});

  const [selectedCallbackParams, setSelectedCallbackParams] = useState<string[]>([]);

  const [showEditCallback, setShowEditCallback] = useState(false);
  const [selectedEditCallback, setSelectedEditCallback] = useState<any | null>(null);
  const [showDeleteConfirmModal, setShowDeleteConfirmModal] = useState(false);
  const [callbackToDelete, setCallbackToDelete] = useState<string | null>(null);

  useEffect(() => {
    if (!accessToken) {
      return;
    }
    getCallbackConfigsCall(accessToken)
      .then((data) => {
        setCallbackConfigs(data || []);
      })
      .catch((error) => {
        NotificationsManager.fromBackend("Failed to load callback configs: " + parseErrorMessage(error));
      });
  }, [accessToken]);

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

  const handleSelectedCallbackChange = (callbackName: string) => {
    setSelectedCallback(callbackName);

    const callbackConfig = callbackConfigs.find((config) => config.id === callbackName);

    if (callbackConfig?.dynamic_params) {
      setSelectedCallbackParams(Object.keys(callbackConfig.dynamic_params));
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
              <LoggingCallbacksTable
                callbacks={callbacks}
                availableCallbacks={allCallbacks}
                onAdd={() => setShowAddCallbacksModal(true)}
                onEdit={(cb) => {
                  setSelectedEditCallback(cb);
                  setShowEditCallback(true);
                }}
                onDelete={(cb) => handleDeleteCallback(cb.name)}
                onTest={async (cb) => {
                  try {
                    await serviceHealthCheck(accessToken, cb.name);
                    NotificationsManager.success("Health check triggered");
                  } catch (error) {
                    NotificationsManager.fromBackend(parseErrorMessage(error));
                  }
                }}
              />
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
        open={showAddCallbacksModal}
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
          <FormItem label="Callback" name="callback" rules={[{ required: true, message: "Please select a callback" }]}>
            <Select
              placeholder="Choose a logging callback..."
              size="large"
              className="w-full"
              showSearch
              filterOption={(input, option) => {
                return (option?.value?.toString() ?? "").toLowerCase().includes(input.toLowerCase());
              }}
              onChange={(value) => {
                handleSelectedCallbackChange(value);
              }}
            >
              {callbackConfigs.map((callbackConfig) => {
                const logo = callbackConfig.logo;
                const logoSrc =
                  logo && (logo.includes("/") || logo.startsWith("data:") || logo.startsWith("http"))
                    ? logo
                    : `${assetsLogoFolder}${logo}`;

                return (
                  <SelectItem key={callbackConfig.id} value={callbackConfig.id}>
                    <div className="flex items-center space-x-3 py-1">
                      <div className="w-6 h-6 flex items-center justify-center">
                        {/* eslint-disable-next-line @next/next/no-img-element */}
                        <img
                          src={logoSrc}
                          alt={`${callbackConfig.displayName} logo`}
                          className="w-6 h-6 rounded object-contain"
                          onError={(e) => {
                            e.currentTarget.style.display = "none";
                          }}
                        />
                      </div>
                      <span className="font-medium text-gray-900">{callbackConfig.displayName}</span>
                    </div>
                  </SelectItem>
                );
              })}
            </Select>
          </FormItem>

          {selectedCallbackParams && selectedCallbackParams.length > 0 && (
            <div className="space-y-4 mt-6 p-4 bg-gray-50 rounded-lg border">
              {selectedCallbackParams.map((param) => {
                const callbackConfig = callbackConfigs.find((config) => config.id === selectedCallback);
                const paramConfig = callbackConfig?.dynamic_params?.[param] || {};
                const paramType = paramConfig.type || "text";
                const fieldLabel =
                  paramConfig.ui_name || param.replace(/_/g, " ").replace(/\b\w/g, (l) => l.toUpperCase());
                const isRequired = paramConfig.required || false;

                return (
                  <FormItem
                    label={<span className="text-sm font-medium text-gray-700">{fieldLabel} </span>}
                    name={param}
                    key={param}
                    className="mb-4"
                    rules={
                      isRequired
                        ? [
                            {
                              required: true,
                              message: `Please enter the ${fieldLabel.toLowerCase()}`,
                            },
                          ]
                        : undefined
                    }
                  >
                    {paramType === "password" ? (
                      <Input.Password
                        size="large"
                        placeholder={`Enter your ${fieldLabel.toLowerCase()}`}
                        className="w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                      />
                    ) : paramType === "number" ? (
                      <Input
                        type="number"
                        size="large"
                        placeholder={`Enter ${fieldLabel.toLowerCase()}`}
                        className="w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                        min={0}
                        max={1}
                        step={0.1}
                      />
                    ) : (
                      <Input
                        size="large"
                        placeholder={`Enter your ${fieldLabel.toLowerCase()}`}
                        className="w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                      />
                    )}
                  </FormItem>
                );
              })}
            </div>
          )}

          <div className="flex justify-end space-x-3 pt-6 mt-6 border-t border-gray-200">
            <Button2
              onClick={() => {
                setShowAddCallbacksModal(false);
                setSelectedCallback(null);
                setSelectedCallbackParams([]);
                addForm.resetFields();
              }}
            >
              Cancel
            </Button2>
            <Button2 htmlType="submit">Add Callback</Button2>
          </div>
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
