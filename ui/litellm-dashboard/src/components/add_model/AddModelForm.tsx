import { useProviderFields } from "@/app/(dashboard)/hooks/providers/useProviderFields";
import { useGuardrails } from "@/app/(dashboard)/hooks/guardrails/useGuardrails";
import { useTags } from "@/app/(dashboard)/hooks/tags/useTags";
import { all_admin_roles, isUserTeamAdminForAnyTeam } from "@/utils/roles";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Alert,
  AlertDescription,
  AlertTitle,
} from "@/components/ui/alert";
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Info, Loader2, X } from "lucide-react";
import React, { useEffect, useMemo, useState } from "react";
import {
  Controller,
  FormProvider,
  UseFormReturn,
  useFormContext,
  useWatch,
} from "react-hook-form";
import TeamDropdown from "../common_components/team_dropdown";
import type { Team } from "../key_team_helpers/key_list";
import {
  type CredentialItem,
  type ProviderCreateInfo,
  modelAvailableCall,
} from "../networking";
import { Providers, providerLogoMap } from "../provider_info_helpers";
import { ProviderLogo } from "../molecules/models/ProviderLogo";
import AdvancedSettings from "./advanced_settings";
import ConditionalPublicModelName from "./conditional_public_model_name";
import LiteLLMModelNameField from "./litellm_model_name";
import ConnectionErrorDisplay from "./model_connection_test";
import ProviderSpecificFields from "./provider_specific_fields";
import { TEST_MODES } from "./add_model_modes";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import type { UploadProps } from "./add_model_upload_types";

export interface AddModelFormValues {
  team_id?: string;
  custom_llm_provider?: string;
  model?: string | string[];
  custom_model_name?: string;
  model_name?: string;
  mode?: string;
  litellm_credential_name?: string | null;
  model_access_group?: string[];
  model_mappings?: { public_name: string; litellm_model: string }[];
  // Advanced settings
  custom_pricing?: boolean;
  pricing_model?: "per_token" | "per_second";
  input_cost_per_token?: string | number | null;
  output_cost_per_token?: string | number | null;
  input_cost_per_second?: string | number | null;
  vector_store_ids?: string[];
  guardrails?: string[];
  tags?: string[];
  cache_control?: boolean;
  cache_control_injection_points?: {
    location: "message";
    role?: string;
    index?: number | null;
  }[];
  use_in_pass_through?: boolean;
  litellm_extra_params?: string;
  model_info_params?: string;
  // Allow provider-specific credential fields to be stored under arbitrary keys.
  [key: string]: unknown;
}

interface AddModelFormProps {
  form: UseFormReturn<AddModelFormValues>;
  handleOk: () => Promise<void>;
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

const AddModelForm: React.FC<AddModelFormProps> = ({
  form,
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
  const [isResultModalVisible, setIsResultModalVisible] =
    useState<boolean>(false);
  const [isTestingConnection, setIsTestingConnection] =
    useState<boolean>(false);
  const [connectionTestId, setConnectionTestId] = useState<string>("");

  const { accessToken, userRole, premiumUser, userId } = useAuthorized();
  const {
    data: providerMetadata,
    isLoading: isProviderMetadataLoading,
    error: providerMetadataError,
  } = useProviderFields();
  const { data: guardrailsData } = useGuardrails();
  const guardrailsList = guardrailsData?.guardrails.map(
    (g) => g.guardrail_name,
  );
  const { data: tagsList } = useTags();

  const handleTestConnection = async () => {
    setIsTestingConnection(true);
    setConnectionTestId(`test-${Date.now()}`);
    setIsResultModalVisible(true);
  };

  const [isTeamOnly, setIsTeamOnly] = useState<boolean>(false);
  const [modelAccessGroups, setModelAccessGroups] = useState<string[]>([]);
  const [teamAdminSelectedTeam, setTeamAdminSelectedTeam] = useState<
    string | null
  >(null);

  useEffect(() => {
    const fetchModelAccessGroups = async () => {
      const response = await modelAvailableCall(
        accessToken,
        "",
        "",
        false,
        null,
        true,
        true,
      );
      setModelAccessGroups(
        response["data"].map((model: { id: string }) => model.id),
      );
    };
    fetchModelAccessGroups();
  }, [accessToken]);

  const sortedProviderMetadata: ProviderCreateInfo[] = useMemo(() => {
    if (!providerMetadata) {
      return [];
    }
    return [...providerMetadata].sort((a, b) =>
      a.provider_display_name.localeCompare(b.provider_display_name),
    );
  }, [providerMetadata]);

  const providerMetadataErrorText = providerMetadataError
    ? providerMetadataError instanceof Error
      ? providerMetadataError.message
      : "Failed to load providers"
    : null;

  const isAdmin = all_admin_roles.includes(userRole);
  const isTeamAdmin = isUserTeamAdminForAnyTeam(teams, userId);

  const onSubmit = form.handleSubmit(async () => {
    await handleOk().then(() => {
      setTeamAdminSelectedTeam(null);
    });
  });

  return (
    <FormProvider {...form}>
      <h2 className="text-2xl font-semibold mb-4">Add Model</h2>

      <Card className="p-6">
        <form onSubmit={onSubmit}>
          {isTeamAdmin && !isAdmin && (
            <>
              <TeamSelectField
                onTeamSelected={setTeamAdminSelectedTeam}
                required
              />
              {!teamAdminSelectedTeam && (
                <Alert className="mb-4">
                  <Info className="h-4 w-4" />
                  <AlertTitle>Team Selection Required</AlertTitle>
                  <AlertDescription>
                    As a team admin, you need to select your team first before
                    adding models.
                  </AlertDescription>
                </Alert>
              )}
            </>
          )}

          {(isAdmin || (isTeamAdmin && teamAdminSelectedTeam)) && (
            <>
              <div className="grid grid-cols-24 gap-2 mb-4 items-start">
                <Label
                  className="col-span-10 pt-2"
                  title="E.g. OpenAI, Azure OpenAI, Anthropic, Bedrock, etc."
                >
                  Provider <span className="text-destructive">*</span>
                </Label>
                <div className="col-span-14 space-y-1">
                  <Controller
                    control={form.control}
                    name="custom_llm_provider"
                    rules={{ required: "Required" }}
                    render={({ field }) => (
                      <Select
                        value={(field.value as string) || ""}
                        onValueChange={(value) => {
                          field.onChange(value);
                          setSelectedProvider(value as Providers);
                          setProviderModelsFn(value as Providers);
                          form.setValue("model", []);
                          form.setValue("model_name", undefined);
                        }}
                      >
                        <SelectTrigger data-testid="provider-select">
                          <SelectValue
                            placeholder={
                              isProviderMetadataLoading
                                ? "Loading providers..."
                                : "Select a provider"
                            }
                          />
                        </SelectTrigger>
                        <SelectContent>
                          {providerMetadataErrorText &&
                            sortedProviderMetadata.length === 0 && (
                              <SelectItem key="__error" value="__error">
                                {providerMetadataErrorText}
                              </SelectItem>
                            )}
                          {sortedProviderMetadata.map((providerInfo) => {
                            const displayName =
                              providerInfo.provider_display_name;
                            const providerKey = providerInfo.provider;
                            // referenced via data-label (search hint) only
                            void providerLogoMap[displayName];

                            return (
                              <SelectItem
                                key={providerKey}
                                value={providerKey}
                              >
                                <div className="flex items-center space-x-2">
                                  <ProviderLogo
                                    provider={providerKey}
                                    className="w-5 h-5"
                                  />
                                  <span>{displayName}</span>
                                </div>
                              </SelectItem>
                            );
                          })}
                        </SelectContent>
                      </Select>
                    )}
                  />
                  {form.formState.errors.custom_llm_provider?.message && (
                    <p className="text-sm text-destructive">
                      {String(
                        form.formState.errors.custom_llm_provider.message,
                      )}
                    </p>
                  )}
                </div>
              </div>

              <LiteLLMModelNameField
                selectedProvider={selectedProvider}
                providerModels={providerModels}
                getPlaceholder={getPlaceholder}
              />

              <ConditionalPublicModelName />

              <div className="grid grid-cols-24 gap-2 mb-1 items-start">
                <Label className="col-span-10 pt-2">Mode</Label>
                <div className="col-span-14">
                  <Controller
                    control={form.control}
                    name="mode"
                    render={({ field }) => (
                      <Select
                        value={(field.value as string) || testMode}
                        onValueChange={(value) => {
                          field.onChange(value);
                          setTestMode(value);
                        }}
                      >
                        <SelectTrigger>
                          <SelectValue placeholder="Select a mode" />
                        </SelectTrigger>
                        <SelectContent>
                          {TEST_MODES.map((opt) => (
                            <SelectItem
                              key={opt.value}
                              value={opt.value as string}
                            >
                              {opt.label}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    )}
                  />
                </div>
              </div>

              <div className="grid grid-cols-24 gap-2 mb-5">
                <div className="col-span-10" />
                <p className="col-span-14 text-sm mt-1">
                  <strong>Optional</strong> - LiteLLM endpoint to use when
                  health checking this model{" "}
                  <a
                    href="https://docs.litellm.ai/docs/proxy/health#health"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-primary hover:text-primary/80 underline"
                  >
                    Learn more
                  </a>
                </p>
              </div>

              <div className="mb-4">
                <p className="text-sm text-muted-foreground mb-2">
                  Either select existing credentials OR enter new provider
                  credentials below
                </p>
              </div>

              <div className="grid grid-cols-24 gap-2 mb-4 items-start">
                <Label className="col-span-10 pt-2">Existing Credentials</Label>
                <div className="col-span-14">
                  <Controller
                    control={form.control}
                    name="litellm_credential_name"
                    defaultValue={null}
                    render={({ field }) => (
                      <Select
                        value={(field.value as string) || "__none"}
                        onValueChange={(v) =>
                          field.onChange(v === "__none" ? null : v)
                        }
                      >
                        <SelectTrigger>
                          <SelectValue placeholder="Select or search for existing credentials" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="__none">None</SelectItem>
                          {credentials.map((credential) => (
                            <SelectItem
                              key={credential.credential_name}
                              value={credential.credential_name}
                            >
                              {credential.credential_name}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    )}
                  />
                </div>
              </div>

              <CredentialsGate
                selectedProvider={selectedProvider}
                uploadProps={uploadProps}
              />

              <div className="flex items-center my-4">
                <div className="flex-grow border-t border-border"></div>
                <span className="px-4 text-muted-foreground text-sm">
                  Additional Model Info Settings
                </span>
                <div className="flex-grow border-t border-border"></div>
              </div>

              {(isAdmin || !isTeamAdmin) && (
                <div className="grid grid-cols-24 gap-2 mb-4 items-center">
                  <Label
                    className="col-span-10"
                    title="Only use this model + credential combination for this team. Useful when teams want to onboard their own OpenAI keys."
                  >
                    Team-BYOK Model
                  </Label>
                  <div className="col-span-14">
                    <span
                      title={
                        !premiumUser
                          ? "This is an enterprise-only feature. Upgrade to premium to restrict model+credential combinations to a specific team."
                          : ""
                      }
                    >
                      <Switch
                        checked={isTeamOnly}
                        onCheckedChange={(checked) => {
                          setIsTeamOnly(!!checked);
                          if (!checked) {
                            form.setValue("team_id", undefined);
                          }
                        }}
                        disabled={!premiumUser}
                      />
                    </span>
                  </div>
                </div>
              )}

              {isTeamOnly && (isAdmin || !isTeamAdmin) && (
                <TeamSelectField
                  onTeamSelected={() => {}}
                  required={isTeamOnly && !isAdmin}
                  disabled={!premiumUser}
                  tooltip="Only keys for this team will be able to call this model."
                />
              )}

              {isAdmin && (
                <div className="grid grid-cols-24 gap-2 mb-4 items-start">
                  <Label
                    className="col-span-10 pt-2"
                    title="Use model access groups to give users access to select models, and add new ones to the group over time."
                  >
                    Model Access Group
                  </Label>
                  <div className="col-span-14">
                    <Controller
                      control={form.control}
                      name="model_access_group"
                      defaultValue={[]}
                      render={({ field }) => (
                        <AccessGroupTagInput
                          value={(field.value as string[]) ?? []}
                          onChange={field.onChange}
                          options={modelAccessGroups}
                        />
                      )}
                    />
                  </div>
                </div>
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
            <a
              href="https://github.com/BerriAI/litellm/issues"
              title="Get help on our github"
              target="_blank"
              rel="noopener noreferrer"
              className="text-primary hover:text-primary/80 underline"
            >
              Need Help?
            </a>
            <div className="space-x-2">
              <Button
                type="button"
                variant="outline"
                data-testid="test-connect-btn"
                onClick={handleTestConnection}
                disabled={isTestingConnection}
              >
                {isTestingConnection && (
                  <Loader2 className="h-4 w-4 animate-spin mr-2" />
                )}
                Test Connect
              </Button>
              <Button data-testid="add-model-btn" type="submit">
                Add Model
              </Button>
            </div>
          </div>
        </form>
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
        <DialogContent className="sm:max-w-[700px]">
          <DialogHeader>
            <DialogTitle>Connection Test Results</DialogTitle>
          </DialogHeader>
          {isResultModalVisible && (
            <ConnectionErrorDisplay
              key={connectionTestId}
              formValues={form.getValues()}
              accessToken={accessToken}
              testMode={testMode}
              modelName={
                (form.getValues("model_name") as string) ||
                ((Array.isArray(form.getValues("model"))
                  ? (form.getValues("model") as string[])[0]
                  : (form.getValues("model") as string)) as string)
              }
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
              <X className="h-4 w-4 mr-1" />
              Close
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </FormProvider>
  );
};

function TeamSelectField({
  onTeamSelected,
  required,
  disabled,
  tooltip,
}: {
  onTeamSelected: (teamId: string | null) => void;
  required?: boolean;
  disabled?: boolean;
  tooltip?: string;
}) {
  const { control, formState } = useFormContext<AddModelFormValues>();
  const error = (formState.errors as Record<string, { message?: string }>).team_id;

  return (
    <div className="grid grid-cols-24 gap-2 mb-4 items-start">
      <Label
        className="col-span-10 pt-2"
        title={tooltip ?? "Select the team for which you want to add this model"}
      >
        Select Team
        {required && <span className="text-destructive ml-1">*</span>}
      </Label>
      <div className="col-span-14 space-y-1">
        <Controller
          control={control}
          name="team_id"
          rules={required ? { required: "Please select a team to continue" } : {}}
          render={({ field }) => (
            <TeamDropdown
              value={field.value as string | undefined}
              onChange={(value) => {
                field.onChange(value);
                onTeamSelected(value || null);
              }}
              disabled={disabled}
            />
          )}
        />
        {error?.message && (
          <p className="text-sm text-destructive">{String(error.message)}</p>
        )}
      </div>
    </div>
  );
}

function CredentialsGate({
  selectedProvider,
  uploadProps,
}: {
  selectedProvider: Providers;
  uploadProps: UploadProps;
}) {
  const { control } = useFormContext<AddModelFormValues>();
  const credentialName = useWatch({ control, name: "litellm_credential_name" });
  if (credentialName) return null;
  return (
    <>
      <div className="flex items-center my-4">
        <div className="flex-grow border-t border-border"></div>
        <span className="px-4 text-muted-foreground text-sm">OR</span>
        <div className="flex-grow border-t border-border"></div>
      </div>
      <ProviderSpecificFields
        selectedProvider={selectedProvider}
        uploadProps={uploadProps}
      />
    </>
  );
}

function AccessGroupTagInput({
  value,
  onChange,
  options,
}: {
  value: string[];
  onChange: (next: string[]) => void;
  options: string[];
}) {
  const [input, setInput] = React.useState("");
  const remaining = options.filter((o) => !value.includes(o));

  const addValue = (next: string) => {
    const trimmed = next.trim();
    if (!trimmed || value.includes(trimmed)) return;
    onChange([...value, trimmed]);
  };

  return (
    <div className="space-y-2">
      <div className="flex gap-2">
        <Input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Type a group name and press Enter"
          onKeyDown={(e) => {
            if (e.key === "Enter" || e.key === ",") {
              e.preventDefault();
              addValue(input);
              setInput("");
            }
          }}
        />
        <Select
          value=""
          onValueChange={(v) => {
            if (v) addValue(v);
          }}
        >
          <SelectTrigger className="w-48">
            <SelectValue placeholder="Pick existing" />
          </SelectTrigger>
          <SelectContent>
            {remaining.length === 0 ? (
              <div className="py-2 px-3 text-sm text-muted-foreground">
                No options available
              </div>
            ) : (
              remaining.map((opt) => (
                <SelectItem key={opt} value={opt}>
                  {opt}
                </SelectItem>
              ))
            )}
          </SelectContent>
        </Select>
      </div>
      {value.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {value.map((v) => (
            <Badge
              key={v}
              variant="secondary"
              className="flex items-center gap-1"
            >
              {v}
              <button
                type="button"
                onClick={() => onChange(value.filter((s) => s !== v))}
                className="inline-flex items-center justify-center rounded-full hover:bg-muted-foreground/20"
                aria-label={`Remove ${v}`}
              >
                <X size={12} />
              </button>
            </Badge>
          ))}
        </div>
      )}
    </div>
  );
}

export default AddModelForm;
