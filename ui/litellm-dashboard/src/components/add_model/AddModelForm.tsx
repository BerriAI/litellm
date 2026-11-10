import { useProviderFields } from "@/app/(dashboard)/hooks/providers/useProviderFields";
import { useGuardrails } from "@/app/(dashboard)/hooks/guardrails/useGuardrails";
import { useTags } from "@/app/(dashboard)/hooks/tags/useTags";
import { all_admin_roles, isUserTeamAdminForAnyTeam } from "@/utils/roles";
import { modelCreationScope } from "@/utils/modelPermissions";
import { Switch } from "@/components/ui/switch";
import { Field, FieldLabel } from "@/components/ui/field";
import { Card, CardContent } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { SearchSelect, type SearchSelectOption } from "@/components/shared/SearchSelect";
import { credentialOptions } from "@/components/shared/credentialOptions";
import { SimpleTooltip } from "@/components/ui/tooltip";
import { Info } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/shared/Alert";
import { Button } from "@/components/ui/button";
import React, { useEffect, useMemo, useState } from "react";
import { FormProvider, useWatch, type UseFormReturn } from "react-hook-form";
import { Trans, useTranslation } from "react-i18next";
import TeamDropdown from "../common_components/team_dropdown";
import { requiredRule } from "../common_components/formRules";
import { labelWithHint } from "@/components/shared/form/LabelWithHint";
import {
  MountedFormField,
  MountedFormProvider,
  type MountRegistry,
  type MountedFormValues,
} from "../common_components/MountedFormField";
import type { Team } from "../key_team_helpers/key_list";
import { type CredentialItem, type ProviderCreateInfo, credentialCreateCall, modelAvailableCall } from "../networking";
import CredentialModal from "../model_add/CredentialModal";
import { federatedProviderOf } from "../model_add/credential_federation";
import { buildCredential, type CredentialSubmission } from "../model_add/credential_form_helpers";
import { ProviderLogo } from "../molecules/models/ProviderLogo";
import AccessGroupTagsCombobox from "./AccessGroupTagsCombobox";
import AdvancedSettings from "./advanced_settings";
import ConditionalPublicModelName from "./conditional_public_model_name";
import LiteLLMModelNameField from "./litellm_model_name";
import ConnectionErrorDisplay from "./model_connection_test";
import ProviderSpecificFields from "./provider_specific_fields";
import { authTypesFor } from "./provider_auth_types";
import { TEST_MODES } from "./add_model_modes";
import DecisionModelNote from "./DecisionModelNote";
import { type DecisionCatalog } from "@/lib/decisionModels";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { credentialsKeys } from "@/app/(dashboard)/hooks/credentials/useCredentials";
import { extractProxyErrorMessage } from "@/lib/http/client";
import { toast } from "@/lib/toast";
import { isProxyAdminRole } from "@/utils/roles";
import { useQueryClient } from "@tanstack/react-query";
import { Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";

interface AddModelFormProps {
  form: UseFormReturn<MountedFormValues>; // For the Add Model tab
  registry: MountRegistry;
  mountedValues: () => MountedFormValues;
  handleOk: () => Promise<boolean>;
  selectedProvider: string | null;
  setSelectedProvider: (provider: string | null) => void;
  providerModels: string[];
  getPlaceholder: (provider: string) => string;
  showAdvancedSettings: boolean;
  setShowAdvancedSettings: (show: boolean) => void;
  teams: Team[] | null;
  credentials: CredentialItem[];
  decisionCatalog: DecisionCatalog;
}

const connectionTestModelName = (values: MountedFormValues): string | undefined => {
  const named = values.model_name || values.model;
  if (Array.isArray(named)) {
    return named.join(", ");
  }
  return typeof named === "string" ? named : undefined;
};

const AddModelForm: React.FC<AddModelFormProps> = ({
  form,
  registry,
  mountedValues,
  handleOk,
  selectedProvider,
  setSelectedProvider,
  providerModels,
  getPlaceholder,
  showAdvancedSettings,
  setShowAdvancedSettings,
  teams,
  credentials,
  decisionCatalog,
}) => {
  const { t } = useTranslation();
  const [testMode, setTestMode] = useState<string>("chat");
  const [isResultModalVisible, setIsResultModalVisible] = useState<boolean>(false);
  const [isTestingConnection, setIsTestingConnection] = useState<boolean>(false);
  // Using a unique ID to force the ConnectionErrorDisplay to remount and run a fresh test
  const [connectionTestId, setConnectionTestId] = useState<string>("");

  const { accessToken, userRole, premiumUser, userId, isViewOnly } = useAuthorized();
  const {
    data: providerMetadata,
    isLoading: isProviderMetadataLoading,
    error: providerMetadataError,
  } = useProviderFields();
  const { data: guardrailsData } = useGuardrails();
  const guardrailsList = guardrailsData?.guardrails.map((g) => g.guardrail_name);
  const { data: tagsList } = useTags();
  const selectedCredentialName = useWatch({ control: form.control, name: "litellm_credential_name" });
  const pickedProvider = useWatch({ control: form.control, name: "custom_llm_provider" });
  const queryClient = useQueryClient();
  const [isCredentialModalOpen, setIsCredentialModalOpen] = useState(false);
  const [credentialModalAuthTypeId, setCredentialModalAuthTypeId] = useState<string | undefined>();
  const [selectedAuthTypeId, setSelectedAuthTypeId] = useState("");
  const canCreateCredential = isProxyAdminRole(userRole ?? "");
  const [isFederatedCredentialModalOpen, setIsFederatedCredentialModalOpen] = useState(false);
  const canCreateFederatedCredential =
    isProxyAdminRole(userRole ?? "") && federatedProviderOf(selectedProvider) !== null;
  const selectedAuthType = authTypesFor(selectedProvider).find(({ id }) => id === selectedAuthTypeId);

  const handleCreateCredential = async (submission: CredentialSubmission) => {
    const credential = buildCredential(submission, submission.credential_values);
    try {
      await credentialCreateCall(accessToken, credential);
    } catch (error) {
      toast.error(extractProxyErrorMessage(error));
      return;
    }
    toast.success(t("addModel.addModelForm.credentialAdded"));
    setIsCredentialModalOpen(false);
    setIsFederatedCredentialModalOpen(false);
    await queryClient.invalidateQueries({ queryKey: credentialsKeys.all });
    form.setValue("litellm_credential_name", credential.credential_name, { shouldDirty: true });
  };

  const openCredentialModal = (authTypeId?: string) => {
    setCredentialModalAuthTypeId(authTypeId);
    setIsCredentialModalOpen(true);
  };

  const rejectMissingCredential = (): boolean => {
    const missing = selectedAuthType?.credentialOnly === true && !selectedCredentialName;
    if (missing) {
      toast.error(t("addModel.addModelForm.credentialRequiredForAuthType"));
    }
    return missing;
  };

  const handleSubmit = async () => (rejectMissingCredential() ? false : handleOk());

  const handleTestConnection = async () => {
    if (rejectMissingCredential() || !(await form.trigger(registry.mountedNames() as string[]))) {
      return;
    }
    setIsTestingConnection(true);
    setConnectionTestId(`test-${Date.now()}`);
    setIsResultModalVisible(true);
  };

  const [isTeamOnly, setIsTeamOnly] = useState<boolean>(false);
  const [modelAccessGroups, setModelAccessGroups] = useState<string[]>([]);
  // Team admin specific state
  const [teamAdminSelectedTeam, setTeamAdminSelectedTeam] = useState<string | null>(null);

  useEffect(() => {
    const fetchModelAccessGroups = async () => {
      const response = await modelAvailableCall(accessToken, "", "", false, null, true, true);
      setModelAccessGroups(response["data"].map((model: any) => model["id"]));
    };
    fetchModelAccessGroups();
  }, [accessToken]);

  const sortedProviderMetadata: ProviderCreateInfo[] = useMemo(() => {
    if (!providerMetadata) {
      return [];
    }
    return [...providerMetadata].sort((a, b) => a.provider_display_name.localeCompare(b.provider_display_name));
  }, [providerMetadata]);

  const providerOptions: SearchSelectOption[] = useMemo(
    () =>
      sortedProviderMetadata.map((providerInfo) => ({
        label: providerInfo.provider_display_name,
        value: providerInfo.provider,
        icon: <ProviderLogo provider={providerInfo.provider} className="w-5 h-5" />,
      })),
    [sortedProviderMetadata],
  );

  const pickedLitellmProvider = sortedProviderMetadata.find(
    (providerInfo) => providerInfo.provider === pickedProvider,
  )?.litellm_provider;

  const credentialSelectOptions: SearchSelectOption[] = useMemo(() => credentialOptions(credentials), [credentials]);

  const applyProviderSelection = (provider: string | null) => {
    setSelectedProvider(provider);
    form.setValue("model", []);
    form.setValue("model_name", undefined);
  };

  const providerMetadataErrorText = providerMetadataError
    ? providerMetadataError instanceof Error
      ? providerMetadataError.message
      : t("addModel.addModelForm.failedToLoadProviders")
    : null;

  const isAdmin = all_admin_roles.includes(userRole);
  const isTeamAdmin = isUserTeamAdminForAnyTeam(teams, userId ?? "");
  // Same owner the Auto-Routers tab uses, so the two creation forms cannot disagree about
  // who has to name a team. This form is only reachable when creation is allowed at all.
  const createScope = modelCreationScope(
    { userRole, userID: userId, isViewOnly },
    { teams, disabledForInternalUsers: false },
  );
  const requiresTeamScope = createScope === "team-required";

  return (
    <>
      <h2 className="mb-4 text-2xl font-semibold text-foreground">{t("addModel.addModelForm.title")}</h2>

      <Card>
        <CardContent>
          <FormProvider {...form}>
            <MountedFormProvider value={{ control: form.control, registry }}>
              <form
                onSubmit={(event) => {
                  event.preventDefault();
                  void handleSubmit().then((submitted) => {
                    if (submitted) {
                      setTeamAdminSelectedTeam(null);
                    }
                  });
                }}
              >
                <>
                  {requiresTeamScope && (
                    <>
                      <MountedFormField
                        label={labelWithHint(
                          t("addModel.addModelForm.selectTeam"),
                          t("addModel.addModelForm.teamScopeHint"),
                        )}
                        name="team_id"
                        required
                        rules={{ validate: { required: requiredRule(t("addModel.addModelForm.teamScopeRequired")) } }}
                        className="mb-4"
                      >
                        {(control) => (
                          <TeamDropdown
                            value={control.value as string | undefined}
                            onChange={(value) => {
                              control.onChange(value);
                              setTeamAdminSelectedTeam(value);
                            }}
                          />
                        )}
                      </MountedFormField>
                      {!teamAdminSelectedTeam && (
                        <Alert variant="info" className="mb-4">
                          <Info />
                          <AlertTitle>{t("addModel.addModelForm.teamSelectionRequiredTitle")}</AlertTitle>
                          <AlertDescription>
                            {t("addModel.addModelForm.teamSelectionRequiredDescription")}
                          </AlertDescription>
                        </Alert>
                      )}
                    </>
                  )}
                  {(isAdmin || (isTeamAdmin && teamAdminSelectedTeam)) && (
                    <>
                      <MountedFormField
                        label={labelWithHint(
                          t("addModel.addModelForm.provider"),
                          t("addModel.addModelForm.providerHint"),
                        )}
                        name="custom_llm_provider"
                        required
                        rules={{ validate: { required: requiredRule(t("addModel.addModelForm.required")) } }}
                        className="mb-4"
                      >
                        {(control) => (
                          <SearchSelect
                            inputId={control.id}
                            options={providerOptions}
                            emptyText={providerMetadataErrorText ?? t("addModel.addModelForm.noProvidersFound")}
                            placeholder={
                              isProviderMetadataLoading
                                ? t("addModel.addModelForm.loadingProviders")
                                : t("addModel.addModelForm.selectProvider")
                            }
                            value={typeof control.value === "string" ? control.value : null}
                            onValueChange={(value) => {
                              control.onChange(value);
                              applyProviderSelection(value);
                            }}
                          />
                        )}
                      </MountedFormField>
                      <LiteLLMModelNameField
                        selectedProvider={selectedProvider}
                        providerModels={providerModels}
                        getPlaceholder={getPlaceholder}
                      />
                      <DecisionModelNote catalog={decisionCatalog} litellmProvider={pickedLitellmProvider} />

                      {/* Conditionally Render "Public Model Name" */}
                      <ConditionalPublicModelName />

                      {/* Select Mode */}
                      <MountedFormField label={t("addModel.addModelForm.mode")} name="mode" className="mb-1">
                        {(control) => (
                          <Select
                            items={TEST_MODES.map((mode) => ({ value: mode.value, label: t(mode.labelKey) }))}
                            value={(control.value as string | undefined) ?? null}
                            onValueChange={(value: string | null) => {
                              control.onChange(value);
                              setTestMode(value ?? "");
                            }}
                          >
                            <SelectTrigger
                              id={control.id}
                              className="w-full"
                              aria-label={t("addModel.addModelForm.mode")}
                            >
                              <SelectValue />
                            </SelectTrigger>
                            <SelectContent>
                              {TEST_MODES.map((mode) => (
                                <SelectItem key={mode.value} value={mode.value}>
                                  {t(mode.labelKey)}
                                </SelectItem>
                              ))}
                            </SelectContent>
                          </Select>
                        )}
                      </MountedFormField>
                      <div className="grid grid-cols-12">
                        <div className="col-span-5" />
                        <div className="col-span-5">
                          <p className="text-sm mb-5 mt-1">
                            <Trans
                              i18nKey="addModel.addModelForm.healthCheckModeHint"
                              components={{
                                strong: <strong />,
                                docs: (
                                  <a
                                    href="https://docs.litellm.ai/docs/proxy/health#health"
                                    target="_blank"
                                    rel="noreferrer"
                                    className="text-primary hover:underline"
                                  />
                                ),
                              }}
                            />
                          </p>
                        </div>
                      </div>

                      {/* Credentials */}
                      <div className="mb-4">
                        <span className="text-sm text-muted-foreground">
                          {t("addModel.addModelForm.credentialsChoiceHint")}
                        </span>
                      </div>

                      <MountedFormField
                        label={t("addModel.addModelForm.existingCredentials")}
                        name="litellm_credential_name"
                        defaultValue={null}
                        className="mb-4"
                      >
                        {(control) => (
                          <SearchSelect
                            inputId={control.id}
                            placeholder={t("addModel.addModelForm.existingCredentialsPlaceholder")}
                            options={credentialSelectOptions}
                            value={(control.value as string | null | undefined) ?? ""}
                            onValueChange={(value) => control.onChange(value === "" ? null : value)}
                          />
                        )}
                      </MountedFormField>

                      {/* Only show provider specific fields if no credentials selected */}
                      {!selectedCredentialName && (
                        <>
                          <div className="flex items-center my-4">
                            <div className="grow border-t border-border"></div>
                            <span className="px-4 text-muted-foreground text-sm">{t("addModel.addModelForm.or")}</span>
                            <div className="grow border-t border-border"></div>
                          </div>
                          <ProviderSpecificFields
                            selectedProvider={selectedProvider}
                            context="model"
                            onAuthTypeChange={setSelectedAuthTypeId}
                            onCreateCredential={canCreateCredential ? openCredentialModal : undefined}
                          />
                          {canCreateFederatedCredential && (
                            <div className="mb-4 flex flex-col items-start gap-2">
                              <span className="text-sm text-muted-foreground">
                                {t("addModel.addModelForm.federationHint")}
                              </span>
                              <Button
                                type="button"
                                variant="outline"
                                onClick={() => setIsFederatedCredentialModalOpen(true)}
                              >
                                {t("addModel.addModelForm.useFederation")}
                              </Button>
                            </div>
                          )}
                        </>
                      )}
                      <div className="flex items-center my-4">
                        <div className="grow border-t border-border"></div>
                        <span className="px-4 text-muted-foreground text-sm">
                          {t("addModel.addModelForm.additionalModelInfoSettings")}
                        </span>
                        <div className="grow border-t border-border"></div>
                      </div>
                      {/* Team-only Model Switch - Only show for proxy admins, not team admins */}
                      {(isAdmin || !isTeamAdmin) && (
                        <Field className="mb-4">
                          <FieldLabel>
                            {labelWithHint(
                              t("addModel.addModelForm.teamByokModel"),
                              t("addModel.addModelForm.teamByokModelHint"),
                            )}
                          </FieldLabel>
                          <SimpleTooltip
                            content={!premiumUser ? t("addModel.addModelForm.teamByokEnterpriseOnly") : ""}
                            side="top"
                          >
                            <span className="inline-flex">
                              <Switch
                                checked={isTeamOnly}
                                onCheckedChange={(checked) => {
                                  setIsTeamOnly(checked);
                                  if (!checked) {
                                    form.setValue("team_id", undefined);
                                  }
                                }}
                                disabled={!premiumUser}
                                aria-label={t("addModel.addModelForm.teamByokModel")}
                              />
                            </span>
                          </SimpleTooltip>
                        </Field>
                      )}

                      {/* Conditional Team Selection */}
                      {isTeamOnly && !requiresTeamScope && (
                        <MountedFormField
                          label={labelWithHint(
                            t("addModel.addModelForm.selectTeam"),
                            t("addModel.addModelForm.teamOnlyHint"),
                          )}
                          name="team_id"
                          className="mb-4"
                          required={isTeamOnly && !isAdmin}
                          rules={
                            isTeamOnly && !isAdmin
                              ? { validate: { required: requiredRule(t("addModel.addModelForm.teamOnlyRequired")) } }
                              : undefined
                          }
                        >
                          {(control) => (
                            <TeamDropdown
                              value={control.value as string | undefined}
                              onChange={control.onChange}
                              disabled={!premiumUser}
                            />
                          )}
                        </MountedFormField>
                      )}
                      {isAdmin && (
                        <>
                          <MountedFormField
                            label={labelWithHint(
                              t("addModel.addModelForm.modelAccessGroup"),
                              t("addModel.addModelForm.modelAccessGroupHint"),
                            )}
                            name="model_access_group"
                            className="mb-4"
                          >
                            {(control) => (
                              <AccessGroupTagsCombobox
                                id={control.id}
                                value={control.value as string[] | undefined}
                                onChange={control.onChange}
                                options={modelAccessGroups}
                                ariaInvalid={control["aria-invalid"] ? true : undefined}
                                ariaDescribedBy={control["aria-describedby"]}
                              />
                            )}
                          </MountedFormField>
                        </>
                      )}
                      <AdvancedSettings
                        showAdvancedSettings={showAdvancedSettings}
                        setShowAdvancedSettings={setShowAdvancedSettings}
                        teams={teams}
                        guardrailsList={guardrailsList || []}
                        tagsList={tagsList || {}}
                        accessToken={accessToken || ""}
                      />
                    </>
                  )}
                  <div className="flex justify-between items-center mb-4">
                    <SimpleTooltip content={t("addModel.addModelForm.helpTooltip")}>
                      <a
                        href="https://github.com/BerriAI/litellm/issues"
                        className="text-sm text-primary hover:underline"
                      >
                        {t("addModel.addModelForm.needHelp")}
                      </a>
                    </SimpleTooltip>
                    <div className="space-x-2">
                      <Button
                        variant="outline"
                        data-testid="test-connect-btn"
                        onClick={handleTestConnection}
                        disabled={isTestingConnection}
                        aria-busy={isTestingConnection}
                      >
                        {t("addModel.addModelForm.testConnect")}
                      </Button>
                      <Button data-testid="add-model-btn" type="submit">
                        {t("addModel.addModelForm.submit")}
                      </Button>
                    </div>
                  </div>
                </>
              </form>
            </MountedFormProvider>
          </FormProvider>
        </CardContent>
      </Card>

      {isCredentialModalOpen && (
        <CredentialModal
          open
          mode="add"
          initialProvider={selectedProvider}
          initialAuthTypeId={credentialModalAuthTypeId}
          providerLocked
          onCancel={() => setIsCredentialModalOpen(false)}
          onSubmit={handleCreateCredential}
        />
      )}
      {isFederatedCredentialModalOpen && (
        <CredentialModal
          open
          mode="add"
          initialProvider={selectedProvider}
          initialAuthMethod="federation"
          providerLocked
          onCancel={() => setIsFederatedCredentialModalOpen(false)}
          onSubmit={handleCreateCredential}
        />
      )}
      {/* Test Connection Results Modal */}
      <Dialog
        open={isResultModalVisible}
        onOpenChange={(open) => {
          if (!open) {
            setIsResultModalVisible(false);
            setIsTestingConnection(false);
          }
        }}
      >
        <DialogContent className="max-h-[calc(100dvh-2rem)] overflow-y-auto sm:max-w-[700px]">
          <DialogHeader>
            <DialogTitle>{t("addModel.addModelForm.connectionTestResults")}</DialogTitle>
          </DialogHeader>
          {/* Only render the ConnectionErrorDisplay when modal is visible and we have a test ID */}
          {isResultModalVisible && (
            <ConnectionErrorDisplay
              // The key prop tells React to create a fresh component instance when it changes
              key={connectionTestId}
              formValues={mountedValues()}
              accessToken={accessToken}
              testMode={testMode}
              modelName={connectionTestModelName(form.getValues())}
              onClose={() => {
                setIsResultModalVisible(false);
                setIsTestingConnection(false);
              }}
              onTestComplete={() => setIsTestingConnection(false)}
            />
          )}
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => {
                setIsResultModalVisible(false);
                setIsTestingConnection(false);
              }}
            >
              {t("addModel.addModelForm.close")}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
};

export default AddModelForm;
