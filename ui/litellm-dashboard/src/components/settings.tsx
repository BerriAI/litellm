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
import { Logo } from "@/components/molecules/logo/Logo";
import NotificationsManager from "./molecules/notifications_manager";

const { Title, Paragraph } = Typography;

import FormItem from "antd/es/form/FormItem";
import AlertingSettings from "./alerting/alerting_settings";
import CloudZeroCostTracking from "./CloudZeroCostTracking/CloudZeroCostTracking";
import DeleteResourceModal from "./common_components/DeleteResourceModal";
import {
  credentialDeleteCall,
  deleteCallback,
  getCallbackConfigsCall,
  getCallbacksCall,
  serviceHealthCheck,
  setCallbacksCall,
} from "./networking";
import { LoggingCallbacksTable } from "./Settings/LoggingAndAlerts/LoggingCallbacks/LoggingCallbacksTable";
import { AlertingObject, CredentialAccess, ResolvedScope } from "./Settings/LoggingAndAlerts/LoggingCallbacks/types";
import { useCredentials } from "@/app/(dashboard)/hooks/credentials/useCredentials";
import { useTeams } from "@/app/(dashboard)/hooks/teams/useTeams";
import { useOrganizations } from "@/app/(dashboard)/hooks/organizations/useOrganizations";
import EditLoggingCredentialModal from "./logging_credentials/EditLoggingCredentialModal";
import AccessControlFields from "./logging_credentials/AccessControlFields";
import { loggingExportersOf } from "./logging_credentials/loggingExportersOf";
import {
  backendLabel,
  createLoggingCredential,
  LOGGING_BACKEND_IDS,
  NON_CALLBACK_LOGGING_IDS,
} from "./logging_credentials/loggingCredentialApi";
import { LOGGING_DESTINATION_BACKENDS } from "./logging_credentials/loggingDestinationFields";
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

const assetsLogoFolder = "/ui/assets/logos/";

export const backendCallbackLogoSrc = (logo: string | null | undefined): string | undefined => {
  if (!logo) return undefined;
  if (logo.includes("/") || logo.startsWith("data:") || logo.startsWith("http")) return logo;
  return `${assetsLogoFolder}${logo}`;
};

interface DynamicParamsFieldsProps {
  params: string[];
  callbackConfigs: any[];
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
                className="w-full rounded-md border-gray-300 shadow-xs focus:border-blue-500 focus:ring-blue-500"
              />
            ) : paramType === "number" ? (
              <Input
                type="number"
                size="large"
                placeholder={`Enter ${fieldLabel.toLowerCase()}`}
                className="w-full rounded-md border-gray-300 shadow-xs focus:border-blue-500 focus:ring-blue-500"
                min={0}
                max={1}
                step={0.1}
              />
            ) : (
              <Input
                size="large"
                placeholder={`Enter your ${fieldLabel.toLowerCase()}`}
                className="w-full rounded-md border-gray-300 shadow-xs focus:border-blue-500 focus:ring-blue-500"
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
  callbackConfigs: any[];
  selectedCallback: string | null;
  onCallbackChange: (value: string) => void;
  disabled?: boolean;
}

export const CallbackSelector: React.FC<CallbackSelectorProps> = ({
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
          return (
            <SelectItem key={callbackConfig.id} value={callbackConfig.id}>
              <div className="flex items-center space-x-3 py-1">
                <div className="w-6 h-6 flex items-center justify-center">
                  <Logo
                    src={backendCallbackLogoSrc(callbackConfig.logo)}
                    label={callbackConfig.displayName}
                    className="w-6 h-6 rounded-sm object-contain"
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
  callbackConfigs: any[],
  fallbackVariables?: Record<string, any>,
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
const buildCallbackPayload = (formValues: Record<string, any>, callbackName: string) => {
  return {
    environment_variables: formValues,
    litellm_settings: {
      success_callback: [callbackName],
    },
  };
};

const Settings: React.FC<SettingsPageProps> = ({ accessToken, userRole, userID, premiumUser }) => {
  const [callbacks, setCallbacks] = useState<AlertingObject[]>([]);
  const [isLoadingCallbacks, setIsLoadingCallbacks] = useState(true);
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
  const [callbackToDelete, setCallbackToDelete] = useState<any | null>(null);
  const [isUpdatingCallback, setIsUpdatingCallback] = useState(false);
  const [isAddingCallback, setIsAddingCallback] = useState(false);
  const [isDeletingCallback, setIsDeletingCallback] = useState(false);

  // OTEL trace destinations are credentials tagged credential_type=logging; they share
  // the one Active Logging Callbacks table as rows alongside config callbacks.
  const { data: credentialData, refetch: refetchCredentials } = useCredentials();
  const { data: teamsData } = useTeams();
  const { data: orgsData } = useOrganizations();
  const [editAccessFor, setEditAccessFor] = useState<{ name: string; access?: CredentialAccess } | null>(null);
  // access for the destination branch of the unified Add modal
  const [addAccess, setAddAccess] = useState<CredentialAccess>({});
  // explicit global/default (auto_enable) opt-in for the destination branch
  const [addAutoEnable, setAddAutoEnable] = useState(false);
  const addingDestination = selectedCallback != null && LOGGING_BACKEND_IDS.has(selectedCallback);
  const addingDestinationFields = LOGGING_DESTINATION_BACKENDS.find((b) => b.id === selectedCallback)?.fields ?? [];

  const teamAlias = (id: string): string => {
    const t = (teamsData ?? []).find((team) => team.team_id === id);
    return t?.team_alias || id;
  };
  const orgAlias = (id: string): string => {
    const o = (orgsData ?? []).find((org) => org.organization_id === id);
    return o?.organization_alias || id;
  };

  // For each destination, the Scope column reflects BOTH directions:
  //  (a) destination-side credential_info.access (global/teams/orgs on the credential)
  //  (b) identity-side metadata.logging_exporters on each team/org that lists this destination
  // The resolver unions them at request time; the column unions them at render time.
  const resolveScope = (destinationName: string, access?: CredentialAccess): ResolvedScope => {
    const teams = new Set<string>();
    const orgs = new Set<string>();
    let global = access?.global === true;
    for (const teamId of access?.teams ?? []) teams.add(teamAlias(teamId));
    for (const orgId of access?.orgs ?? []) orgs.add(orgAlias(orgId));
    for (const team of teamsData ?? []) {
      const exporters = loggingExportersOf(team);
      if (exporters.includes(destinationName)) {
        teams.add(team.team_alias || team.team_id);
      }
    }
    for (const org of orgsData ?? []) {
      const exporters = loggingExportersOf(org);
      if (exporters.includes(destinationName)) {
        orgs.add(org.organization_alias || org.organization_id);
      }
    }
    return { global, teams: Array.from(teams), orgs: Array.from(orgs) };
  };

  const destinationRows: AlertingObject[] = (credentialData?.credentials ?? [])
    .filter((c) => c.credential_info?.credential_type === "logging")
    .map((c) => ({
      name: c.credential_name,
      variables: {} as AlertingObject["variables"],
      credentialName: c.credential_name,
      destinationLabel: c.credential_info?.host
        ? `${backendLabel(c.credential_info?.description)} · ${c.credential_info.host}`
        : backendLabel(c.credential_info?.description),
      access: c.credential_info?.access,
      autoEnable: c.credential_info?.auto_enable === true,
      resolvedScope: resolveScope(c.credential_name, c.credential_info?.access),
    }));

  const handleDeleteDestination = async (name: string) => {
    if (!accessToken) return;
    try {
      await credentialDeleteCall(accessToken, name);
      NotificationsManager.success("Logging destination deleted");
      refetchCredentials();
    } catch (error) {
      NotificationsManager.fromBackend(parseErrorMessage(error));
    }
  };

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
      editForm.setFieldsValue({
        ...normalized,
        callback: selectedEditCallback.name,
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

  useEffect(() => {
    const fetchCallbacks = async () => {
      if (!accessToken || !userRole || !userID) {
        setIsLoadingCallbacks(false);
        return;
      }
      try {
        const data = await getCallbacksCall(accessToken, userID, userRole);
        setCallbacks(data.callbacks);
        setAllCallbacks(data.available_callbacks);

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
      } finally {
        setIsLoadingCallbacks(false);
      }
    };
    fetchCallbacks();
  }, [accessToken, userRole, userID]);

  const isAlertOn = (alertName: string) => {
    return activeAlerts && activeAlerts.includes(alertName);
  };

  // Shared handler for callback form submission
  const handleCallbackSubmit = async (formValues: Record<string, any>, callbackName: string, isEdit: boolean) => {
    if (!accessToken) {
      return;
    }

    if (isEdit) {
      setIsUpdatingCallback(true);
    } else {
      setIsAddingCallback(true);
    }

    const payload = buildCallbackPayload(formValues, callbackName);

    try {
      await setCallbacksCall(accessToken, payload);
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

      // Refresh the callbacks list
      if (userID && userRole) {
        const updatedData = await getCallbacksCall(accessToken, userID, userRole);
        setCallbacks(updatedData.callbacks);
      }
    } catch (error) {
      NotificationsManager.fromBackend(error);
    } finally {
      if (isEdit) {
        setIsUpdatingCallback(false);
      } else {
        setIsAddingCallback(false);
      }
    }
  };

  const updateCallbackCall = async (formValues: Record<string, any>) => {
    if (!selectedEditCallback) {
      return;
    }
    await handleCallbackSubmit(formValues, selectedEditCallback.name, true);
  };

  const addNewCallbackCall = async (formValues: Record<string, any>) => {
    const new_callback = formValues?.callback;
    if (!new_callback) {
      return;
    }
    if (LOGGING_BACKEND_IDS.has(new_callback) && accessToken) {
      const backendDef = LOGGING_DESTINATION_BACKENDS.find((b) => b.id === new_callback);
      const fields = backendDef?.fields ?? [];
      const values = Object.fromEntries(
        fields.filter((f) => formValues[f.name]).map((f) => [f.name, formValues[f.name]]),
      );
      const host = backendDef ? formValues[backendDef.hostField] : undefined;
      const hasAccess = addAccess.global || addAccess.teams?.length || addAccess.orgs?.length;
      try {
        await createLoggingCredential(accessToken, {
          credentialName: formValues.credential_name,
          backend: new_callback,
          values,
          host,
          access: hasAccess ? addAccess : undefined,
          autoEnable: addAutoEnable,
        });
        NotificationsManager.success("Logging destination created");
        refetchCredentials();
        setShowAddCallbacksModal(false);
        setSelectedCallback(null);
        setAddAccess({});
        setAddAutoEnable(false);
        addForm.resetFields();
      } catch (error) {
        NotificationsManager.fromBackend(parseErrorMessage(error));
      }
      return;
    }
    await handleCallbackSubmit(formValues, new_callback, false);
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

  const handleDeleteCallback = (callback: any) => {
    setCallbackToDelete(callback);
    setShowDeleteConfirmModal(true);
  };

  const confirmDeleteCallback = async () => {
    if (!callbackToDelete || !accessToken) {
      return;
    }

    try {
      setIsDeletingCallback(true);
      await deleteCallback(accessToken, callbackToDelete.name);
      NotificationsManager.success(`Callback ${callbackToDelete.name} deleted successfully`);

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
    } finally {
      setIsDeletingCallback(false);
    }
  };

  if (!accessToken) {
    return null;
  }

  return (
    <div className="mx-4">
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
                callbacks={[...callbacks.filter((c) => !NON_CALLBACK_LOGGING_IDS.has(c.name)), ...destinationRows]}
                availableCallbacks={allCallbacks}
                isLoading={isLoadingCallbacks}
                onAdd={() => setShowAddCallbacksModal(true)}
                onEdit={(cb) => {
                  setSelectedEditCallback(cb);
                  setShowEditCallback(true);
                }}
                onEditAccess={(cb) =>
                  cb.credentialName && setEditAccessFor({ name: cb.credentialName, access: cb.access })
                }
                onDelete={(cb) =>
                  cb.credentialName ? handleDeleteDestination(cb.credentialName) : handleDeleteCallback(cb)
                }
                onTest={async (cb) => {
                  try {
                    await serviceHealthCheck(accessToken, cb.name);
                    NotificationsManager.success("Health check triggered");
                  } catch (error) {
                    NotificationsManager.fromBackend(parseErrorMessage(error));
                  }
                }}
              />
              {accessToken && (
                <EditLoggingCredentialModal
                  accessToken={accessToken}
                  credentialName={editAccessFor?.name ?? null}
                  access={editAccessFor?.access}
                  open={editAccessFor != null}
                  onClose={() => setEditAccessFor(null)}
                  onSaved={() => refetchCredentials()}
                />
              )}
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
          setAddAccess({});
          setAddAutoEnable(false);
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
            callbackConfigs={[
              ...callbackConfigs.filter((c: { id: string }) => !NON_CALLBACK_LOGGING_IDS.has(c.id)),
              ...LOGGING_DESTINATION_BACKENDS.map((b) => ({ id: b.id, displayName: b.label, logo: "" })),
            ]}
            selectedCallback={selectedCallback}
            onCallbackChange={handleSelectedCallbackChange}
          />

          {addingDestination ? (
            <div className="space-y-4 mt-6 p-4 bg-gray-50 rounded-lg border">
              <FormItem
                label={<span className="text-sm font-medium text-gray-700">Name</span>}
                name="credential_name"
                rules={[{ required: true, message: "Please enter a name" }]}
              >
                <Input size="large" placeholder="e.g. langfuse-eu" />
              </FormItem>
              {addingDestinationFields.map((f) => (
                <FormItem
                  key={f.name}
                  label={<span className="text-sm font-medium text-gray-700">{f.label}</span>}
                  name={f.name}
                  rules={
                    f.optional ? undefined : [{ required: true, message: `Please enter the ${f.label.toLowerCase()}` }]
                  }
                >
                  {f.type === "password" ? (
                    <Input.Password size="large" placeholder={f.placeholder} />
                  ) : (
                    <Input size="large" placeholder={f.placeholder} />
                  )}
                </FormItem>
              ))}
              <AccessControlFields value={addAccess} onChange={setAddAccess} />
              <Form.Item
                label="Auto-enable for all requests"
                tooltip="When on, every request exports its traces to this destination automatically, without being named on a key, team, or org. The explicit global default; replaces relying on Global to auto-enable."
              >
                <Switch checked={addAutoEnable} onChange={setAddAutoEnable} />
              </Form.Item>
            </div>
          ) : (
            <DynamicParamsFields
              params={selectedCallbackParams}
              callbackConfigs={callbackConfigs}
              selectedCallback={selectedCallback}
            />
          )}

          <div className="flex justify-end space-x-3 pt-6 mt-6 border-t border-gray-200">
            <Button2
              onClick={() => {
                setShowAddCallbacksModal(false);
                setSelectedCallback(null);
                setSelectedCallbackParams([]);
                setAddAccess({});
                setAddAutoEnable(false);
                addForm.resetFields();
              }}
              disabled={isAddingCallback}
            >
              Cancel
            </Button2>
            <Button2 htmlType="submit" loading={isAddingCallback} disabled={isAddingCallback}>
              {isAddingCallback ? "Adding..." : "Add"}
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
              disabled={isUpdatingCallback}
            >
              Cancel
            </Button2>
            <Button2
              onClick={() => {
                editForm.submit();
              }}
              loading={isUpdatingCallback}
              disabled={isUpdatingCallback}
            >
              {isUpdatingCallback ? "Saving..." : "Save Changes"}
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
          { label: "Mode", value: callbackToDelete?.mode || "success" },
        ]}
        onCancel={() => {
          setShowDeleteConfirmModal(false);
          setCallbackToDelete(null);
        }}
        onOk={confirmDeleteCallback}
        confirmLoading={isDeletingCallback}
      />
    </div>
  );
};

export default Settings;
