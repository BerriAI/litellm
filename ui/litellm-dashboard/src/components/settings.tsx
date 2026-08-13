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

import { Button as Button2, Form, Input, Modal, Select } from "antd";
import EmailSettings from "./email_settings";
import { Logo } from "@/components/molecules/logo/Logo";
import NotificationsManager from "./molecules/notifications_manager";

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
import { canReadCredentialsRole, isProxyAdminRole } from "@/utils/roles";
import { useTeams } from "@/app/(dashboard)/hooks/teams/useTeams";
import { useOrganizations } from "@/app/(dashboard)/hooks/organizations/useOrganizations";
import EditLoggingCredentialModal from "./logging_credentials/EditLoggingCredentialModal";
import AccessControlFields from "./logging_credentials/AccessControlFields";
import {
  backendLabel,
  createLoggingCredential,
  DESTINATION_OPTION_PREFIX,
} from "./logging_credentials/loggingCredentialApi";
import { LOGGING_DESTINATION_BACKENDS } from "./logging_credentials/loggingDestinationFields";
import { parseErrorMessage } from "./shared/errorUtils";
interface SettingsPageProps {
  accessToken: string | null;
  userRole: string | null;
  userID: string | null;
  premiumUser: boolean;
  isViewOnly?: boolean;
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

const Settings: React.FC<SettingsPageProps> = ({ accessToken, userRole, userID, premiumUser, isViewOnly = false }) => {
  const [callbacks, setCallbacks] = useState<AlertingObject[]>([]);
  const [isLoadingCallbacks, setIsLoadingCallbacks] = useState(true);
  const [alerts, setAlerts] = useState<any[]>([]);
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

  // OTEL trace destinations are proxy-admin-managed credentials tagged
  // credential_type=logging; they share the one Active Logging Callbacks table as
  // rows alongside config callbacks. Only a proxy admin (or admin-viewer, read-only)
  // may read them, so non-admins skip the fetch entirely.
  const isProxyAdmin = !isViewOnly && userRole != null && isProxyAdminRole(userRole);
  const { data: credentialData, refetch: refetchCredentials } = useCredentials(canReadCredentialsRole(userRole));
  const { data: teamsData } = useTeams();
  const { data: orgsData } = useOrganizations();
  const [editAccessFor, setEditAccessFor] = useState<{
    name: string;
    access?: CredentialAccess;
    credentialInfo?: Record<string, unknown>;
  } | null>(null);
  // access for the destination branch of the unified Add modal
  const [addAccess, setAddAccess] = useState<CredentialAccess>({});
  const addingDestination = selectedCallback != null && selectedCallback.startsWith(DESTINATION_OPTION_PREFIX);
  const selectedDestinationBackend = addingDestination
    ? selectedCallback.slice(DESTINATION_OPTION_PREFIX.length)
    : null;
  const addingDestinationFields =
    LOGGING_DESTINATION_BACKENDS.find((b) => b.id === selectedDestinationBackend)?.fields ?? [];

  const teamAlias = (id: string): string => {
    const t = (teamsData ?? []).find((team) => team.team_id === id);
    return t?.team_alias || id;
  };
  const orgAlias = (id: string): string => {
    const o = (orgsData ?? []).find((org) => org.organization_id === id);
    return o?.organization_alias || id;
  };

  const asIdList = (value: unknown): string[] =>
    Array.isArray(value) ? value.filter((entry): entry is string => typeof entry === "string") : [];

  const resolveScope = (access?: CredentialAccess): ResolvedScope => ({
    global: access?.global === true,
    teams: asIdList(access?.teams).map(teamAlias),
    orgs: asIdList(access?.orgs).map(orgAlias),
  });

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
      credentialInfo: c.credential_info as Record<string, unknown> | undefined,
      resolvedScope: resolveScope(c.credential_info?.access),
      resolvesToDestination: (c as { resolves_to_destination?: boolean }).resolves_to_destination,
    }));

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
    if (new_callback.startsWith(DESTINATION_OPTION_PREFIX) && accessToken) {
      const backendId = new_callback.slice(DESTINATION_OPTION_PREFIX.length);
      const backendDef = LOGGING_DESTINATION_BACKENDS.find((b) => b.id === backendId);
      const fields = backendDef?.fields ?? [];
      const values = Object.fromEntries(
        fields.filter((f) => formValues[f.name]).map((f) => [f.name, formValues[f.name]]),
      );
      const host = backendDef ? formValues[backendDef.hostField] : undefined;
      const hasAccess = addAccess.global || addAccess.teams?.length || addAccess.orgs?.length;
      try {
        await createLoggingCredential(accessToken, {
          credentialName: formValues.credential_name,
          backend: backendId,
          values,
          host,
          access: hasAccess ? addAccess : undefined,
        });
        NotificationsManager.success("Logging destination created");
        refetchCredentials();
        setShowAddCallbacksModal(false);
        setSelectedCallback(null);
        setAddAccess({});
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
      // A destination row carries a credentialName; it is a logging credential and is
      // deleted (with its stored collector secrets) through the credential endpoint. A
      // plain config callback is deleted through the callback endpoint. Both run only
      // after the same delete confirmation, so a mis-click can't drop either instantly.
      if (callbackToDelete.credentialName) {
        await credentialDeleteCall(accessToken, callbackToDelete.credentialName);
        NotificationsManager.success("Logging destination deleted");
        refetchCredentials();
      } else {
        await deleteCallback(accessToken, callbackToDelete.name);
        NotificationsManager.success(`Callback ${callbackToDelete.name} deleted successfully`);
        // Refresh the callbacks list
        if (userID && userRole) {
          const data = await getCallbacksCall(accessToken, userID, userRole);
          setCallbacks(data.callbacks);
        }
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
                callbacks={[...callbacks, ...destinationRows]}
                availableCallbacks={allCallbacks}
                isLoading={isLoadingCallbacks}
                readOnly={!isProxyAdmin}
                onAdd={() => setShowAddCallbacksModal(true)}
                onEdit={(cb) => {
                  setSelectedEditCallback(cb);
                  setShowEditCallback(true);
                }}
                onEditAccess={(cb) =>
                  cb.credentialName &&
                  setEditAccessFor({ name: cb.credentialName, access: cb.access, credentialInfo: cb.credentialInfo })
                }
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
              {accessToken && (
                <EditLoggingCredentialModal
                  // Remount per destination. Form.useForm() owns a store that outlives the
                  // dialog and initialValues only seeds it on first mount, so reopening for
                  // another credential kept the previous one's scope in the fields while the
                  // title updated -- and Save sends the whole access object, so that stale
                  // scope was written, silently turning a team destination global.
                  // Neither destroyOnHidden nor resetting the fields from an effect fixes
                  // this (the portal's children mount after the effect runs); remounting the
                  // component, and with it the form store, is what works. Verified by A/B.
                  key={editAccessFor?.name ?? "none"}
                  accessToken={accessToken}
                  credentialName={editAccessFor?.name ?? null}
                  access={editAccessFor?.access}
                  credentialInfo={editAccessFor?.credentialInfo}
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
              ...callbackConfigs,
              ...LOGGING_DESTINATION_BACKENDS.map((b) => ({
                id: `${DESTINATION_OPTION_PREFIX}${b.id}`,
                displayName: `${b.label} (scoped destination)`,
                logo: "",
              })),
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
        title={callbackToDelete?.credentialName ? "Delete Destination" : "Delete Callback"}
        message={
          callbackToDelete?.credentialName
            ? "Are you sure you want to delete this trace destination? Its stored collector credentials are deleted with it and traces stop reaching it. This action cannot be undone."
            : "Are you sure you want to delete this callback? This action cannot be undone."
        }
        resourceInformationTitle={callbackToDelete?.credentialName ? "Destination Information" : "Callback Information"}
        resourceInformation={
          // A destination has no mode. Defaulting the shared field to "success" stated a
          // value the row itself renders as "—", so the dialog disagreed with the table.
          callbackToDelete?.credentialName
            ? [
                { label: "Destination Name", value: callbackToDelete?.name },
                { label: "Backend", value: callbackToDelete?.destinationLabel },
              ]
            : [
                { label: "Callback Name", value: callbackToDelete?.name },
                { label: "Mode", value: callbackToDelete?.mode || "success" },
              ]
        }
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
