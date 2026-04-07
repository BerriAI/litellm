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

const { Title } = Typography;

import FormItem from "antd/es/form/FormItem";
import AlertingSettings from "./alerting/alerting_settings";
import CloudZeroCostTracking from "./CloudZeroCostTracking/CloudZeroCostTracking";
import DeleteResourceModal from "./common_components/DeleteResourceModal";
import {
  serviceHealthCheck,
  setCallbacksCall,
} from "./networking";
import { useCallbacks, useCallbackConfigs } from "@/app/(dashboard)/hooks/callbacks/useCallbacks";
import { useUpdateCallback } from "@/app/(dashboard)/hooks/callbacks/useUpdateCallback";
import { useDeleteCallback } from "@/app/(dashboard)/hooks/callbacks/useDeleteCallback";
import { CALLBACK_MODES, LoggingCallbacksTable } from "./Settings/LoggingAndAlerts/LoggingCallbacks/LoggingCallbacksTable";
import { AlertingObject, CallbackConfig } from "./Settings/LoggingAndAlerts/LoggingCallbacks/types";
import { parseErrorMessage } from "./shared/errorUtils";
interface SettingsPageProps {
  accessToken: string | null;
  userRole: string | null;
  userID: string | null;
  premiumUser: boolean;
}

const assetsLogoFolder = "../ui/assets/logos/";

interface DynamicParamsFieldsProps {
  params: string[];
  callbackConfigs: CallbackConfig[];
  selectedCallback: string | null;
}

const DynamicParamsFields: React.FC<DynamicParamsFieldsProps> = ({ params, callbackConfigs, selectedCallback }) => {
  if (!params || params.length === 0) {
    return null;
  }

  return (
    <div className="space-y-4 mt-6 p-4 bg-gray-50 rounded-lg border">
      {params.map((param) => {
        const callbackConfig = callbackConfigs.find((config) => config.id === selectedCallback);
        const paramConfig = callbackConfig?.dynamic_params?.[param] || {};
        const paramType = paramConfig.type || "text";
        const fieldLabel = paramConfig.ui_name || param.replace(/_/g, " ").replace(/\b\w/g, (l) => l.toUpperCase());
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
  );
};

// Shared component for rendering callback selector
interface CallbackSelectorProps {
  callbackConfigs: CallbackConfig[];
  selectedCallback: string | null;
  onCallbackChange: (value: string) => void;
  disabled?: boolean;
}

const CallbackSelector: React.FC<CallbackSelectorProps> = ({
  callbackConfigs,
  selectedCallback,
  onCallbackChange,
  disabled = false,
}) => {
  return (
    <FormItem
      label="Callback"
      name="callback"
      rules={disabled ? undefined : [{ required: true, message: "Please select a callback" }]}
    >
      <Select
        placeholder="Choose a logging callback..."
        size="large"
        className="w-full"
        showSearch
        disabled={disabled}
        value={selectedCallback}
        filterOption={(input, option) => {
          return (option?.value?.toString() ?? "").toLowerCase().includes(input.toLowerCase());
        }}
        onChange={onCallbackChange}
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
  );
};

// Shared helper function to get dynamic params for a callback
const getDynamicParamsForCallback = (
  callbackName: string | null,
  callbackConfigs: CallbackConfig[],
  fallbackVariables?: Record<string, string | null>,
): string[] => {
  if (!callbackName) {
    return fallbackVariables ? Object.keys(fallbackVariables) : [];
  }

  const callbackConfig = callbackConfigs.find((config) => config.id === callbackName);
  if (callbackConfig?.dynamic_params) {
    return Object.keys(callbackConfig.dynamic_params);
  }

  return fallbackVariables ? Object.keys(fallbackVariables) : [];
};

// Shared helper function to build callback payload
const buildCallbackPayload = (formValues: Record<string, any>, callbackName: string, mode: string = "success") => {
  const { callback_mode, callback, ...envVars } = formValues;
  const litellm_settings: Record<string, string[]> = {};

  if (mode === "success") {
    litellm_settings.success_callback = [callbackName];
  } else if (mode === "failure") {
    litellm_settings.failure_callback = [callbackName];
  } else if (mode === "success_and_failure") {
    litellm_settings.success_callback = [callbackName];
    litellm_settings.failure_callback = [callbackName];
  }

  return {
    environment_variables: envVars,
    litellm_settings,
  };
};

const Settings: React.FC<SettingsPageProps> = ({ accessToken, userRole, userID, premiumUser }) => {
  const [addForm] = Form.useForm();
  const [editForm] = Form.useForm();
  const [selectedCallback, setSelectedCallback] = useState<string | null>(null);
  const [catchAllWebhookURL, setCatchAllWebhookURL] = useState<string>("");
  const [alertToWebhooks, setAlertToWebhooks] = useState<Record<string, string>>({});
  const [activeAlerts, setActiveAlerts] = useState<string[]>([]);

  const [showAddCallbacksModal, setShowAddCallbacksModal] = useState(false);
  const [selectedCallbackParams, setSelectedCallbackParams] = useState<string[]>([]);
  const [showEditCallback, setShowEditCallback] = useState(false);
  const [selectedEditCallback, setSelectedEditCallback] = useState<AlertingObject | null>(null);
  const [showDeleteConfirmModal, setShowDeleteConfirmModal] = useState(false);
  const [callbackToDelete, setCallbackToDelete] = useState<AlertingObject | null>(null);

  // React Query hooks for callbacks
  const { data: callbacksData } = useCallbacks();
  const { data: callbackConfigs = [] } = useCallbackConfigs();
  const updateCallbackMutation = useUpdateCallback();
  const deleteCallbackMutation = useDeleteCallback();

  const callbacks = callbacksData?.callbacks ?? [];
  const allCallbacks = callbacksData?.availableCallbacks ?? {};
  const alerts = callbacksData?.alerts ?? [];

  // Derive alerting state from query data
  useEffect(() => {
    if (alerts.length > 0) {
      const alertInfo = alerts[0];
      setCatchAllWebhookURL(alertInfo.variables.SLACK_WEBHOOK_URL);
      setActiveAlerts(alertInfo.active_alerts);
      setAlertToWebhooks(alertInfo.alerts_to_webhook);
    }
  }, [alerts]);

  useEffect(() => {
    if (showEditCallback && selectedEditCallback) {
      const normalized = Object.fromEntries(
        Object.entries(selectedEditCallback.variables || {}).map(([k, v]) => [k, v ?? ""]),
      );
      editForm.setFieldsValue({
        ...normalized,
        callback: selectedEditCallback.name,
        callback_mode: selectedEditCallback.type || "success",
      });
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

  const isAlertOn = (alertName: string) => {
    return activeAlerts && activeAlerts.includes(alertName);
  };

  const handleCallbackSubmit = async (formValues: Record<string, any>, callbackName: string, isEdit: boolean, mode: string = "success") => {
    const payload = buildCallbackPayload(formValues, callbackName, mode);

    updateCallbackMutation.mutate(
      { payload },
      {
        onSuccess: () => {
          NotificationsManager.success(
            isEdit ? "Callback updated successfully" : `Callback ${callbackName} added successfully`,
          );
          if (isEdit) {
            setShowEditCallback(false);
            editForm.resetFields();
            setSelectedEditCallback(null);
          } else {
            setShowAddCallbacksModal(false);
            addForm.resetFields();
            setSelectedCallback(null);
            setSelectedCallbackParams([]);
          }
        },
        onError: (error) => {
          NotificationsManager.fromBackend(error);
        },
      },
    );
  };

  const updateCallbackCall = async (formValues: Record<string, any>) => {
    if (!selectedEditCallback) {
      return;
    }
    const mode = formValues.callback_mode || selectedEditCallback.type || "success";
    await handleCallbackSubmit(formValues, selectedEditCallback.name, true, mode);
  };

  const addNewCallbackCall = async (formValues: Record<string, any>) => {
    const new_callback = formValues?.callback;
    if (!new_callback) {
      return;
    }
    const mode = formValues.callback_mode || "success";
    await handleCallbackSubmit(formValues, new_callback, false, mode);
  };

  const handleSelectedCallbackChange = (callbackName: string) => {
    setSelectedCallback(callbackName);
    const params = getDynamicParamsForCallback(callbackName, callbackConfigs);
    setSelectedCallbackParams(params);
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

  const handleDeleteCallback = (callback: any) => {
    setCallbackToDelete(callback);
    setShowDeleteConfirmModal(true);
  };

  const confirmDeleteCallback = () => {
    if (!callbackToDelete) {
      return;
    }

    deleteCallbackMutation.mutate(callbackToDelete.name, {
      onSuccess: () => {
        NotificationsManager.success(`Callback ${callbackToDelete.name} deleted successfully`);
        setShowDeleteConfirmModal(false);
        setCallbackToDelete(null);
      },
      onError: (error) => {
        NotificationsManager.fromBackend(error);
      },
    });
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
            <Tab value="2">CloudZero Cost Tracking</Tab>
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
                onDelete={(cb) => handleDeleteCallback(cb)}
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
              <div className="p-8">
                <CloudZeroCostTracking />
              </div>
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
          <CallbackSelector
            callbackConfigs={callbackConfigs}
            selectedCallback={selectedCallback}
            onCallbackChange={handleSelectedCallbackChange}
          />

          <FormItem
            label="Callback Mode"
            name="callback_mode"
            initialValue="success"
          >
            <Select size="large" className="w-full">
              {CALLBACK_MODES.map((m) => (
                <SelectItem key={m.value} value={m.value}>
                  {m.label}
                </SelectItem>
              ))}
            </Select>
          </FormItem>

          <DynamicParamsFields
            params={selectedCallbackParams}
            callbackConfigs={callbackConfigs}
            selectedCallback={selectedCallback}
          />

          <div className="flex justify-end space-x-3 pt-6 mt-6 border-t border-gray-200">
            <Button2
              onClick={() => {
                setShowAddCallbacksModal(false);
                setSelectedCallback(null);
                setSelectedCallbackParams([]);
                addForm.resetFields();
              }}
              disabled={updateCallbackMutation.isPending}
            >
              Cancel
            </Button2>
            <Button2 htmlType="submit" loading={updateCallbackMutation.isPending} disabled={updateCallbackMutation.isPending}>
              {updateCallbackMutation.isPending ? "Adding..." : "Add Callback"}
            </Button2>
          </div>
        </Form>
      </Modal>

      <Modal
        open={showEditCallback}
        width={800}
        title={"Edit Callback Settings"}
        onCancel={() => {
          setShowEditCallback(false);
          setSelectedEditCallback(null);
          editForm.resetFields();
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
          {selectedEditCallback && (
            <>
              <CallbackSelector
                callbackConfigs={callbackConfigs}
                selectedCallback={selectedEditCallback.name}
                onCallbackChange={() => {}}
                disabled={true}
              />

              <FormItem
                label="Callback Mode"
                name="callback_mode"
                initialValue={selectedEditCallback.type || "success"}
              >
                <Select size="large" className="w-full">
                  {CALLBACK_MODES.map((m) => (
                    <SelectItem key={m.value} value={m.value}>
                      {m.label}
                    </SelectItem>
                  ))}
                </Select>
              </FormItem>

              <DynamicParamsFields
                params={getDynamicParamsForCallback(
                  selectedEditCallback.name,
                  callbackConfigs,
                  selectedEditCallback.variables,
                )}
                callbackConfigs={callbackConfigs}
                selectedCallback={selectedEditCallback.name}
              />
            </>
          )}

          <div className="flex justify-end space-x-3 pt-6 mt-6 border-t border-gray-200">
            <Button2
              onClick={() => {
                setShowEditCallback(false);
                setSelectedEditCallback(null);
                editForm.resetFields();
              }}
              disabled={updateCallbackMutation.isPending}
            >
              Cancel
            </Button2>
            <Button2
              onClick={() => {
                editForm.submit();
              }}
              loading={updateCallbackMutation.isPending}
              disabled={updateCallbackMutation.isPending}
            >
              {updateCallbackMutation.isPending ? "Saving..." : "Save Changes"}
            </Button2>
          </div>
        </Form>
      </Modal>

      <DeleteResourceModal
        isOpen={showDeleteConfirmModal}
        title="Delete Callback"
        message="Are you sure you want to delete this callback? This action cannot be undone."
        resourceInformationTitle="Callback Information"
        resourceInformation={[
          { label: "Callback Name", value: callbackToDelete?.name },
          { label: "Mode", value: callbackToDelete?.type || "success" },
        ]}
        onCancel={() => {
          setShowDeleteConfirmModal(false);
          setCallbackToDelete(null);
        }}
        onOk={confirmDeleteCallback}
        confirmLoading={deleteCallbackMutation.isPending}
      />
    </div>
  );
};

export default Settings;
