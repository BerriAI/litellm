import React, { useEffect, useState } from "react";
import { Controller, FormProvider, useForm, useFormContext } from "react-hook-form";

import { Field, FieldError, FieldLabel } from "@/components/ui/field";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import {
  Combobox,
  ComboboxContent,
  ComboboxEmpty,
  ComboboxInput,
  ComboboxItem,
  ComboboxList,
} from "@/components/ui/combobox";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import EmailSettings from "./email_settings";
import { Logo } from "@/components/molecules/logo/Logo";
import { toast } from "@/lib/toast";

import AlertingSettings from "./alerting/alerting_settings";
import CloudZeroCostTracking from "./CloudZeroCostTracking/CloudZeroCostTracking";
import DeleteResourceModal from "./common_components/DeleteResourceModal";
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

type CallbackFormValues = Record<string, string>;

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
  const { register, formState } = useFormContext<CallbackFormValues>();
  const fieldIdPrefix = React.useId();

  if (!params || params.length === 0) {
    return null;
  }

  return (
    <div className="space-y-4 mt-6 p-4 bg-muted rounded-lg border">
      {params.map((param) => {
        const callbackConfig = callbackConfigs.find((config) => config.id === selectedCallback);
        const paramConfig = callbackConfig?.dynamic_params?.[param] || {};
        const paramType = paramConfig.type || "text";
        const fieldLabel = paramConfig.ui_name || param.replace(/_/g, " ").replace(/\b\w/g, (l) => l.toUpperCase());
        const isRequired = paramConfig.required || false;
        const fieldId = `${fieldIdPrefix}-${param}`;
        const registration = register(
          param,
          isRequired ? { required: `Please enter the ${fieldLabel.toLowerCase()}` } : undefined,
        );

        return (
          <Field key={param} className="mb-4">
            <FieldLabel htmlFor={fieldId}>
              <span className="text-sm font-medium text-foreground">{fieldLabel} </span>
            </FieldLabel>
            {paramType === "password" ? (
              <Input
                id={fieldId}
                type="password"
                placeholder={`Enter your ${fieldLabel.toLowerCase()}`}
                {...registration}
              />
            ) : paramType === "number" ? (
              <Input
                id={fieldId}
                type="number"
                placeholder={`Enter ${fieldLabel.toLowerCase()}`}
                min={0}
                max={1}
                step={0.1}
                {...registration}
              />
            ) : (
              <Input id={fieldId} placeholder={`Enter your ${fieldLabel.toLowerCase()}`} {...registration} />
            )}
            <FieldError errors={[formState.errors[param]]} />
          </Field>
        );
      })}
    </div>
  );
};

interface CallbackConfigOption {
  id: string;
  displayName: string;
  logo?: string | null;
}

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
  const { control } = useFormContext<CallbackFormValues>();
  const inputId = React.useId();
  const selectedConfig = callbackConfigs.find((config) => config.id === selectedCallback) ?? null;

  return (
    <Controller
      control={control}
      name="callback"
      rules={disabled ? undefined : { required: "Please select a callback" }}
      render={({ field, fieldState }) => (
        <Field>
          <FieldLabel htmlFor={inputId}>Callback</FieldLabel>
          <Combobox
            items={callbackConfigs}
            value={selectedConfig}
            onValueChange={(config: CallbackConfigOption | null) => {
              field.onChange(config?.id ?? "");
              onCallbackChange(config?.id ?? "");
            }}
            isItemEqualToValue={(a: CallbackConfigOption, b: CallbackConfigOption) => a.id === b.id}
            itemToStringLabel={(config: CallbackConfigOption) => config.displayName}
            filter={(config: CallbackConfigOption, query: string) =>
              config.id.toLowerCase().includes(query.trim().toLowerCase())
            }
            disabled={disabled}
          >
            <ComboboxInput
              id={inputId}
              placeholder="Choose a logging callback..."
              className="w-full"
              disabled={disabled}
              onBlur={field.onBlur}
              aria-invalid={fieldState.error !== undefined || undefined}
            />
            <ComboboxContent>
              <ComboboxEmpty>No results</ComboboxEmpty>
              <ComboboxList>
                {(callbackConfig: CallbackConfigOption) => (
                  <ComboboxItem key={callbackConfig.id} value={callbackConfig}>
                    <div className="flex items-center space-x-3 py-1">
                      <div className="w-6 h-6 flex items-center justify-center">
                        <Logo
                          src={backendCallbackLogoSrc(callbackConfig.logo)}
                          label={callbackConfig.displayName}
                          className="w-6 h-6 rounded-sm object-contain"
                        />
                      </div>
                      <span className="font-medium text-foreground">{callbackConfig.displayName}</span>
                    </div>
                  </ComboboxItem>
                )}
              </ComboboxList>
            </ComboboxContent>
          </Combobox>
          <FieldError errors={[fieldState.error]} />
        </Field>
      )}
    />
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
  const addForm = useForm<CallbackFormValues>({ shouldUnregister: true });
  const editForm = useForm<CallbackFormValues>({ shouldUnregister: true });
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

  useEffect(() => {
    if (!accessToken) {
      return;
    }
    getCallbackConfigsCall(accessToken)
      .then((data) => {
        setCallbackConfigs(data || []);
      })
      .catch((error) => {
        toast.fromError("Failed to load callback configs: " + parseErrorMessage(error));
      });
  }, [accessToken]);

  useEffect(() => {
    if (showEditCallback && selectedEditCallback) {
      const normalized = Object.fromEntries(
        Object.entries(selectedEditCallback.variables || {}).map(([k, v]) => [k, v ?? ""]),
      );
      editForm.reset({
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
    model_deprecation_warnings: "Model Deprecation Warnings",
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
      toast.success(isEdit ? "Callback updated successfully" : `Callback ${callbackName} added successfully`);

      if (isEdit) {
        setShowEditCallback(false);
        editForm.reset();
        setSelectedEditCallback(null);
      } else {
        setShowAddCallbacksModal(false);
        addForm.reset();
        setSelectedCallback(null);
        setSelectedCallbackParams([]);
      }

      // Refresh the callbacks list
      if (userID && userRole) {
        const updatedData = await getCallbacksCall(accessToken, userID, userRole);
        setCallbacks(updatedData.callbacks);
      }
    } catch (error) {
      toast.fromError(error);
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
    await handleCallbackSubmit(formValues, new_callback, false);
  };

  const handleSelectedCallbackChange = (callbackName: string) => {
    setSelectedCallback(callbackName);
    const params = getDynamicParamsForCallback(callbackName, callbackConfigs);
    setSelectedCallbackParams(params);
  };

  const closeAddCallbackModal = () => {
    setShowAddCallbacksModal(false);
    setSelectedCallback(null);
    setSelectedCallbackParams([]);
  };

  const cancelAddCallback = () => {
    closeAddCallbackModal();
    addForm.reset();
  };

  const closeEditCallbackModal = () => {
    setShowEditCallback(false);
    setSelectedEditCallback(null);
    editForm.reset();
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
      toast.fromError(error);
    }
    toast.success("Alerts updated successfully");
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
      toast.success(`Callback ${callbackToDelete.name} deleted successfully`);

      // Refresh the callbacks list
      if (userID && userRole) {
        const data = await getCallbacksCall(accessToken, userID, userRole);
        setCallbacks(data.callbacks);
      }

      setShowDeleteConfirmModal(false);
      setCallbackToDelete(null);
    } catch (error) {
      console.error("Failed to delete callback:", error);
      toast.fromError(error);
    } finally {
      setIsDeletingCallback(false);
    }
  };

  if (!accessToken) {
    return null;
  }

  return (
    <div className="mx-4">
      <div className="grid grid-cols-1 gap-2 p-8 w-full mt-2">
        <Tabs defaultValue="logging-callbacks">
          <TabsList variant="line">
            <TabsTrigger value="logging-callbacks">Logging Callbacks</TabsTrigger>
            <TabsTrigger value="cloudzero-cost-tracking">CloudZero Cost Tracking</TabsTrigger>
            <TabsTrigger value="alerting-types">Alerting Types</TabsTrigger>
            <TabsTrigger value="alerting-settings">Alerting Settings</TabsTrigger>
            <TabsTrigger value="email-alerts">Email Alerts</TabsTrigger>
          </TabsList>
          <TabsContent value="logging-callbacks" keepMounted>
            <LoggingCallbacksTable
              callbacks={callbacks}
              availableCallbacks={allCallbacks}
              isLoading={isLoadingCallbacks}
              onAdd={() => setShowAddCallbacksModal(true)}
              onEdit={(cb) => {
                setSelectedEditCallback(cb);
                setShowEditCallback(true);
              }}
              onDelete={(cb) => handleDeleteCallback(cb)}
              onTest={async (cb) => {
                try {
                  await serviceHealthCheck(accessToken, cb.name);
                  toast.success("Health check triggered");
                } catch (error) {
                  toast.fromError(parseErrorMessage(error));
                }
              }}
            />
          </TabsContent>
          <TabsContent value="cloudzero-cost-tracking" keepMounted>
            <div className="p-8">
              <CloudZeroCostTracking />
            </div>
          </TabsContent>
          <TabsContent value="alerting-types" keepMounted>
            <Card className="p-6">
              <p className="my-2">
                Alerts are only supported for Slack Webhook URLs. Get your webhook urls from{" "}
                <a href="https://api.slack.com/messaging/webhooks" target="_blank" style={{ color: "blue" }}>
                  here
                </a>
              </p>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead></TableHead>
                    <TableHead></TableHead>
                    <TableHead>Slack Webhook URL</TableHead>
                  </TableRow>
                </TableHeader>

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
                              onCheckedChange={() => handleSwitchChange(key)}
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
                            onCheckedChange={() => handleSwitchChange(key)}
                          />
                        )}
                      </TableCell>
                      <TableCell className="whitespace-normal break-words">
                        <p>{value}</p>
                      </TableCell>
                      <TableCell>
                        <Input
                          name={key}
                          type="password"
                          defaultValue={
                            alertToWebhooks && alertToWebhooks[key]
                              ? alertToWebhooks[key]
                              : (catchAllWebhookURL as string)
                          }
                        />
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
                    toast.success(
                      "Alert test triggered. Test request to slack made - check logs/alerts on slack to verify",
                    );
                  } catch (error) {
                    toast.fromError(parseErrorMessage(error));
                  }
                }}
                className="mx-2"
              >
                Test Alerts
              </Button>
            </Card>
          </TabsContent>
          <TabsContent value="alerting-settings" keepMounted>
            <AlertingSettings accessToken={accessToken} premiumUser={premiumUser} />
          </TabsContent>
          <TabsContent value="email-alerts" keepMounted>
            <EmailSettings accessToken={accessToken} premiumUser={premiumUser} alerts={alerts} />
          </TabsContent>
        </Tabs>
      </div>

      <Dialog open={showAddCallbacksModal} onOpenChange={(open) => !open && closeAddCallbackModal()}>
        <DialogContent className="max-h-[calc(100dvh-2rem)] overflow-y-auto sm:max-w-[800px]">
          <DialogHeader>
            <DialogTitle>Add Logging Callback</DialogTitle>
          </DialogHeader>
          <a
            href="https://docs.litellm.ai/docs/proxy/logging"
            className="mb-8 mt-4"
            target="_blank"
            style={{ color: "blue" }}
          >
            {" "}
            LiteLLM Docs: Logging
          </a>

          <FormProvider {...addForm}>
            <form onSubmit={addForm.handleSubmit(addNewCallbackCall)}>
              <CallbackSelector
                callbackConfigs={callbackConfigs}
                selectedCallback={selectedCallback}
                onCallbackChange={handleSelectedCallbackChange}
              />

              <DynamicParamsFields
                params={selectedCallbackParams}
                callbackConfigs={callbackConfigs}
                selectedCallback={selectedCallback}
              />

              <div className="flex justify-end space-x-3 pt-6 mt-6 border-t border-border">
                <Button type="button" variant="outline" onClick={cancelAddCallback} disabled={isAddingCallback}>
                  Cancel
                </Button>
                <Button type="submit" disabled={isAddingCallback}>
                  {isAddingCallback ? "Adding..." : "Add Callback"}
                </Button>
              </div>
            </form>
          </FormProvider>
        </DialogContent>
      </Dialog>

      <Dialog open={showEditCallback} onOpenChange={(open) => !open && closeEditCallbackModal()}>
        <DialogContent className="max-h-[calc(100dvh-2rem)] overflow-y-auto sm:max-w-[800px]">
          <DialogHeader>
            <DialogTitle>Edit Callback Settings</DialogTitle>
          </DialogHeader>
          <FormProvider {...editForm}>
            <form onSubmit={editForm.handleSubmit(updateCallbackCall)}>
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

              <div className="flex justify-end space-x-3 pt-6 mt-6 border-t border-border">
                <Button type="button" variant="outline" onClick={closeEditCallbackModal} disabled={isUpdatingCallback}>
                  Cancel
                </Button>
                <Button type="submit" disabled={isUpdatingCallback}>
                  {isUpdatingCallback ? "Saving..." : "Save Changes"}
                </Button>
              </div>
            </form>
          </FormProvider>
        </DialogContent>
      </Dialog>

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
