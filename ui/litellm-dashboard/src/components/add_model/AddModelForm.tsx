import { useProviderFields } from "@/app/(dashboard)/hooks/providers/useProviderFields";
import { useGuardrails } from "@/app/(dashboard)/hooks/guardrails/useGuardrails";
import { useTags } from "@/app/(dashboard)/hooks/tags/useTags";
import { all_admin_roles, isUserTeamAdminForAnyTeam } from "@/utils/roles";
import { modelCreationScope } from "@/utils/modelPermissions";
import { Switch } from "@/components/ui/switch";
import { Field, FieldLabel } from "@/components/shared/form/field";
import { Select as AntdSelect, Card, Col, Modal, Row, Tooltip, Typography } from "antd";
import { Info } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/shared/Alert";
import { Button } from "@/components/ui/button";
import type { UploadProps } from "antd/es/upload";
import React, { useEffect, useMemo, useState } from "react";
import { FormProvider, useWatch, type UseFormReturn } from "react-hook-form";
import TeamDropdown from "../common_components/team_dropdown";
import { antdRequired } from "../common_components/antdFormRules";
import { labelWithHint } from "@/components/shared/form/LabelWithHint";
import {
  MountedFormField,
  MountedFormProvider,
  type MountRegistry,
  type MountedFormValues,
} from "../common_components/MountedFormField";
import type { Team } from "../key_team_helpers/key_list";
import { type CredentialItem, type ProviderCreateInfo, modelAvailableCall } from "../networking";
import { Providers } from "../provider_info_helpers";
import { ProviderLogo } from "../molecules/models/ProviderLogo";
import AdvancedSettings from "./advanced_settings";
import ConditionalPublicModelName from "./conditional_public_model_name";
import LiteLLMModelNameField from "./litellm_model_name";
import ConnectionErrorDisplay from "./model_connection_test";
import ProviderSpecificFields from "./provider_specific_fields";
import { TEST_MODES } from "./add_model_modes";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

interface AddModelFormProps {
  form: UseFormReturn<MountedFormValues>; // For the Add Model tab
  registry: MountRegistry;
  mountedValues: () => MountedFormValues;
  handleOk: () => Promise<boolean>;
  selectedProvider: Providers;
  setSelectedProvider: (provider: Providers) => void;
  providerModels: string[];
  setProviderModelsFn: (provider: Providers) => void;
  getPlaceholder: (provider: Providers) => string;
  uploadProps: UploadProps;
  showAdvancedSettings: boolean;
  setShowAdvancedSettings: (show: boolean) => void;
  teams: Team[] | null;
  credentials: CredentialItem[];
}

const connectionTestModelName = (values: MountedFormValues): string | undefined => {
  const named = values.model_name || values.model;
  if (Array.isArray(named)) {
    return named.join(", ");
  }
  return typeof named === "string" ? named : undefined;
};

const { Title, Link } = Typography;

const AddModelForm: React.FC<AddModelFormProps> = ({
  form,
  registry,
  mountedValues,
  handleOk,
  selectedProvider,
  setSelectedProvider,
  providerModels,
  setProviderModelsFn,
  getPlaceholder,
  uploadProps,
  showAdvancedSettings,
  setShowAdvancedSettings,
  teams,
  credentials,
}) => {
  const [testMode, setTestMode] = useState<string>("chat");
  const [isResultModalVisible, setIsResultModalVisible] = useState<boolean>(false);
  const [isTestingConnection, setIsTestingConnection] = useState<boolean>(false);
  // Using a unique ID to force the ConnectionErrorDisplay to remount and run a fresh test
  const [connectionTestId, setConnectionTestId] = useState<string>("");

  const { accessToken, userRole, premiumUser, userId } = useAuthorized();
  const {
    data: providerMetadata,
    isLoading: isProviderMetadataLoading,
    error: providerMetadataError,
  } = useProviderFields();
  const { data: guardrailsData } = useGuardrails();
  const guardrailsList = guardrailsData?.guardrails.map((g) => g.guardrail_name);
  const { data: tagsList } = useTags();
  const selectedCredentialName = useWatch({ control: form.control, name: "litellm_credential_name" });

  const handleTestConnection = async () => {
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

  const providerMetadataErrorText = providerMetadataError
    ? providerMetadataError instanceof Error
      ? providerMetadataError.message
      : "Failed to load providers"
    : null;

  const isAdmin = all_admin_roles.includes(userRole);
  const isTeamAdmin = isUserTeamAdminForAnyTeam(teams, userId);
  // Same owner the Auto-Routers tab uses, so the two creation forms cannot disagree about
  // who has to name a team. This form is only reachable when creation is allowed at all.
  const createScope = modelCreationScope({ userRole, userID: userId }, { teams, disabledForInternalUsers: false });
  const requiresTeamScope = createScope === "team-required";

  return (
    <>
      <Title level={2}>Add Model</Title>

      <Card>
        <FormProvider {...form}>
          <MountedFormProvider value={{ control: form.control, registry }}>
            <form
              onSubmit={(event) => {
                event.preventDefault();
                void handleOk().then((submitted) => {
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
                      label={labelWithHint("Select Team", "Select the team for which you want to add this model")}
                      name="team_id"
                      required
                      rules={{ validate: { required: antdRequired("Please select a team to continue") } }}
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
                        <AlertTitle>Team Selection Required</AlertTitle>
                        <AlertDescription>
                          As a team admin, you need to select your team first before adding models.
                        </AlertDescription>
                      </Alert>
                    )}
                  </>
                )}
                {(isAdmin || (isTeamAdmin && teamAdminSelectedTeam)) && (
                  <>
                    <MountedFormField
                      label={labelWithHint("Provider", "E.g. OpenAI, Azure OpenAI, Anthropic, Bedrock, etc.")}
                      name="custom_llm_provider"
                      required
                      rules={{ validate: { required: antdRequired("Required") } }}
                      className="mb-4"
                    >
                      {(control) => (
                        <AntdSelect
                          id={control.id}
                          virtual={false}
                          showSearch
                          loading={isProviderMetadataLoading}
                          placeholder={isProviderMetadataLoading ? "Loading providers..." : "Select a provider"}
                          optionFilterProp="data-label"
                          value={control.value as string | undefined}
                          onBlur={control.onBlur}
                          onChange={(value) => {
                            control.onChange(value);
                            setSelectedProvider(value as Providers);
                            setProviderModelsFn(value as Providers);
                            form.setValue("model", []);
                            form.setValue("model_name", undefined);
                          }}
                        >
                          {providerMetadataErrorText && sortedProviderMetadata.length === 0 && (
                            <AntdSelect.Option key="__error" value="">
                              {providerMetadataErrorText}
                            </AntdSelect.Option>
                          )}
                          {sortedProviderMetadata.map((providerInfo) => {
                            const displayName = providerInfo.provider_display_name;
                            const providerKey = providerInfo.provider;

                            return (
                              <AntdSelect.Option key={providerKey} value={providerKey} data-label={displayName}>
                                <div className="flex items-center space-x-2">
                                  <ProviderLogo provider={providerKey} className="w-5 h-5" />
                                  <span>{displayName}</span>
                                </div>
                              </AntdSelect.Option>
                            );
                          })}
                        </AntdSelect>
                      )}
                    </MountedFormField>
                    <LiteLLMModelNameField
                      selectedProvider={selectedProvider}
                      providerModels={providerModels}
                      getPlaceholder={getPlaceholder}
                    />

                    {/* Conditionally Render "Public Model Name" */}
                    <ConditionalPublicModelName />

                    {/* Select Mode */}
                    <MountedFormField label="Mode" name="mode" className="mb-1">
                      {(control) => (
                        <AntdSelect
                          id={control.id}
                          style={{ width: "100%" }}
                          value={control.value as string | undefined}
                          onBlur={control.onBlur}
                          onChange={(value) => {
                            control.onChange(value);
                            setTestMode(value);
                          }}
                          options={TEST_MODES}
                        />
                      )}
                    </MountedFormField>
                    <Row>
                      <Col span={10}></Col>
                      <Col span={10}>
                        <p className="text-sm mb-5 mt-1">
                          <strong>Optional</strong> - LiteLLM endpoint to use when health checking this model{" "}
                          <Link href="https://docs.litellm.ai/docs/proxy/health#health" target="_blank">
                            Learn more
                          </Link>
                        </p>
                      </Col>
                    </Row>

                    {/* Credentials */}
                    <div className="mb-4">
                      <Typography.Text className="text-sm text-muted-foreground mb-2">
                        Either select existing credentials OR enter new provider credentials below
                      </Typography.Text>
                    </div>

                    <MountedFormField
                      label="Existing Credentials"
                      name="litellm_credential_name"
                      defaultValue={null}
                      className="mb-4"
                    >
                      {(control) => (
                        <AntdSelect
                          id={control.id}
                          showSearch
                          placeholder="Select or search for existing credentials"
                          optionFilterProp="children"
                          filterOption={(input, option) =>
                            (option?.label ?? "").toLowerCase().includes(input.toLowerCase())
                          }
                          value={control.value as string | null | undefined}
                          onChange={control.onChange}
                          onBlur={control.onBlur}
                          options={[
                            { value: null, label: "None" },
                            ...credentials.map((credential) => ({
                              value: credential.credential_name,
                              label: credential.credential_name,
                            })),
                          ]}
                          allowClear
                        />
                      )}
                    </MountedFormField>

                    {/* Only show provider specific fields if no credentials selected */}
                    {!selectedCredentialName && (
                      <>
                        <div className="flex items-center my-4">
                          <div className="grow border-t border-border"></div>
                          <span className="px-4 text-muted-foreground text-sm">OR</span>
                          <div className="grow border-t border-border"></div>
                        </div>
                        <ProviderSpecificFields selectedProvider={selectedProvider} uploadProps={uploadProps} />
                      </>
                    )}
                    <div className="flex items-center my-4">
                      <div className="grow border-t border-border"></div>
                      <span className="px-4 text-muted-foreground text-sm">Additional Model Info Settings</span>
                      <div className="grow border-t border-border"></div>
                    </div>
                    {/* Team-only Model Switch - Only show for proxy admins, not team admins */}
                    {(isAdmin || !isTeamAdmin) && (
                      <Field className="mb-4">
                        <FieldLabel>
                          {labelWithHint(
                            "Team-BYOK Model",
                            "Only use this model + credential combination for this team. Useful when teams want to onboard their own OpenAI keys.",
                          )}
                        </FieldLabel>
                        <Tooltip
                          title={
                            !premiumUser
                              ? "This is an enterprise-only feature. Upgrade to premium to restrict model+credential combinations to a specific team."
                              : ""
                          }
                          placement="top"
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
                              aria-label="Team-BYOK Model"
                            />
                          </span>
                        </Tooltip>
                      </Field>
                    )}

                    {/* Conditional Team Selection */}
                    {isTeamOnly && !requiresTeamScope && (
                      <MountedFormField
                        label={labelWithHint("Select Team", "Only keys for this team will be able to call this model.")}
                        name="team_id"
                        className="mb-4"
                        required={isTeamOnly && !isAdmin}
                        rules={
                          isTeamOnly && !isAdmin
                            ? { validate: { required: antdRequired("Please select a team.") } }
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
                            "Model Access Group",
                            "Use model access groups to give users access to select models, and add new ones to the group over time.",
                          )}
                          name="model_access_group"
                          className="mb-4"
                        >
                          {(control) => (
                            <AntdSelect
                              id={control.id}
                              mode="tags"
                              showSearch
                              placeholder="Select existing groups or type to create new ones"
                              optionFilterProp="children"
                              tokenSeparators={[","]}
                              value={control.value as string[] | undefined}
                              onChange={control.onChange}
                              onBlur={control.onBlur}
                              options={modelAccessGroups.map((group) => ({
                                value: group,
                                label: group,
                              }))}
                              maxTagCount="responsive"
                              allowClear
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
                  <Tooltip title="Get help on our github">
                    <Typography.Link href="https://github.com/BerriAI/litellm/issues">Need Help?</Typography.Link>
                  </Tooltip>
                  <div className="space-x-2">
                    <Button
                      variant="outline"
                      data-testid="test-connect-btn"
                      onClick={handleTestConnection}
                      disabled={isTestingConnection}
                      aria-busy={isTestingConnection}
                    >
                      Test Connect
                    </Button>
                    <Button data-testid="add-model-btn" type="submit">
                      Add Model
                    </Button>
                  </div>
                </div>
              </>
            </form>
          </MountedFormProvider>
        </FormProvider>
      </Card>

      {/* Test Connection Results Modal */}
      <Modal
        title="Connection Test Results"
        open={isResultModalVisible}
        onCancel={() => {
          setIsResultModalVisible(false);
          setIsTestingConnection(false);
        }}
        footer={[
          <Button
            key="close"
            variant="outline"
            onClick={() => {
              setIsResultModalVisible(false);
              setIsTestingConnection(false);
            }}
          >
            Close
          </Button>,
        ]}
        width={700}
      >
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
      </Modal>
    </>
  );
};

export default AddModelForm;
