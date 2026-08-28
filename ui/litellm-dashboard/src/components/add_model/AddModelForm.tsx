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
import { SimpleTooltip } from "@/components/ui/tooltip";
import { Info } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/shared/Alert";
import { Button } from "@/components/ui/button";
import React, { useEffect, useMemo, useState } from "react";
import { FormProvider, useWatch, type UseFormReturn } from "react-hook-form";
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
import { type CredentialItem, type ProviderCreateInfo, modelAvailableCall } from "../networking";
import { Providers } from "../provider_info_helpers";
import { ProviderLogo } from "../molecules/models/ProviderLogo";
import AccessGroupTagsCombobox from "./AccessGroupTagsCombobox";
import AdvancedSettings from "./advanced_settings";
import ConditionalPublicModelName from "./conditional_public_model_name";
import LiteLLMModelNameField from "./litellm_model_name";
import ConnectionErrorDisplay from "./model_connection_test";
import ProviderSpecificFields from "./provider_specific_fields";
import { TEST_MODES } from "./add_model_modes";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";

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

  const providerOptions: SearchSelectOption[] = useMemo(
    () =>
      sortedProviderMetadata.map((providerInfo) => ({
        label: providerInfo.provider_display_name,
        value: providerInfo.provider,
        icon: <ProviderLogo provider={providerInfo.provider} className="w-5 h-5" />,
      })),
    [sortedProviderMetadata],
  );

  const credentialOptions: SearchSelectOption[] = useMemo(
    () => [
      { label: "None", value: "" },
      ...credentials.map((credential) => ({
        label: credential.credential_name,
        value: credential.credential_name,
      })),
    ],
    [credentials],
  );

  const applyProviderSelection = (provider: Providers) => {
    setSelectedProvider(provider);
    setProviderModelsFn(provider);
    form.setValue("model", []);
    form.setValue("model_name", undefined);
  };

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
      <h2 className="mb-4 text-2xl font-semibold text-foreground">Add Model</h2>

      <Card>
        <CardContent>
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
                        rules={{ validate: { required: requiredRule("Please select a team to continue") } }}
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
                        rules={{ validate: { required: requiredRule("Required") } }}
                        className="mb-4"
                      >
                        {(control) => (
                          <SearchSelect
                            inputId={control.id}
                            options={providerOptions}
                            emptyText={providerMetadataErrorText ?? "No providers found"}
                            placeholder={isProviderMetadataLoading ? "Loading providers..." : "Select a provider"}
                            value={(control.value as string | undefined) ?? ""}
                            onValueChange={(value) => {
                              control.onChange(value);
                              applyProviderSelection(value as Providers);
                            }}
                          />
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
                          <Select
                            items={TEST_MODES}
                            value={(control.value as string | undefined) ?? null}
                            onValueChange={(value: string | null) => {
                              control.onChange(value);
                              setTestMode(value ?? "");
                            }}
                          >
                            <SelectTrigger id={control.id} className="w-full" aria-label="Mode">
                              <SelectValue />
                            </SelectTrigger>
                            <SelectContent>
                              {TEST_MODES.map((mode) => (
                                <SelectItem key={mode.value} value={mode.value}>
                                  {mode.label}
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
                            <strong>Optional</strong> - LiteLLM endpoint to use when health checking this model{" "}
                            <a
                              href="https://docs.litellm.ai/docs/proxy/health#health"
                              target="_blank"
                              rel="noreferrer"
                              className="text-primary hover:underline"
                            >
                              Learn more
                            </a>
                          </p>
                        </div>
                      </div>

                      {/* Credentials */}
                      <div className="mb-4">
                        <span className="text-sm text-muted-foreground">
                          Either select existing credentials OR enter new provider credentials below
                        </span>
                      </div>

                      <MountedFormField
                        label="Existing Credentials"
                        name="litellm_credential_name"
                        defaultValue={null}
                        className="mb-4"
                      >
                        {(control) => (
                          <SearchSelect
                            inputId={control.id}
                            placeholder="Select or search for existing credentials"
                            options={credentialOptions}
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
                            <span className="px-4 text-muted-foreground text-sm">OR</span>
                            <div className="grow border-t border-border"></div>
                          </div>
                          <ProviderSpecificFields selectedProvider={selectedProvider} />
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
                          <SimpleTooltip
                            content={
                              !premiumUser
                                ? "This is an enterprise-only feature. Upgrade to premium to restrict model+credential combinations to a specific team."
                                : ""
                            }
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
                                aria-label="Team-BYOK Model"
                              />
                            </span>
                          </SimpleTooltip>
                        </Field>
                      )}

                      {/* Conditional Team Selection */}
                      {isTeamOnly && !requiresTeamScope && (
                        <MountedFormField
                          label={labelWithHint(
                            "Select Team",
                            "Only keys for this team will be able to call this model.",
                          )}
                          name="team_id"
                          className="mb-4"
                          required={isTeamOnly && !isAdmin}
                          rules={
                            isTeamOnly && !isAdmin
                              ? { validate: { required: requiredRule("Please select a team.") } }
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
                    <SimpleTooltip content="Get help on our github">
                      <a
                        href="https://github.com/BerriAI/litellm/issues"
                        className="text-sm text-primary hover:underline"
                      >
                        Need Help?
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
        </CardContent>
      </Card>

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
            <DialogTitle>Connection Test Results</DialogTitle>
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
            {" "}
            <Button
              variant="outline"
              onClick={() => {
                setIsResultModalVisible(false);
                setIsTestingConnection(false);
              }}
            >
              Close
            </Button>
            , ]
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
};

export default AddModelForm;
