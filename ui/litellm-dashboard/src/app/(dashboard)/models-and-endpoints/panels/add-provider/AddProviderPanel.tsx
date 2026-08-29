"use client";

import React from "react";
import { useForm, FormProvider } from "react-hook-form";
import { useQueryClient } from "@tanstack/react-query";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { toast } from "@/lib/toast";
import { useProviderFields } from "@/app/(dashboard)/hooks/providers/useProviderFields";
import { useCredentials } from "@/app/(dashboard)/hooks/credentials/useCredentials";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import {
  MountedFormProvider,
  projectMountedValues,
  useMountRegistry,
  type MountedFormValues,
} from "@/components/common_components/MountedFormField";
import { computeCredentialValuesToDelete } from "@/components/model_add/credential_form_helpers";
import ProviderSpecificFields from "@/components/add_model/provider_specific_fields";
import { ProviderLogo } from "@/components/molecules/models/ProviderLogo";
import { Providers } from "@/components/provider_info_helpers";
import {
  credentialCreateCall,
  credentialUpdateCall,
  discoverProviderModelsCall,
  getCredentialJwksCall,
  listAllModelsCall,
  getCallbacksCall,
  setCallbacksCall,
  createProviderModelCall,
  type AnthropicJwks,
  type DeploymentInfoRow,
  type ProviderCreateInfo,
} from "@/components/networking";
import type { SearchSelectOption } from "@/components/shared/SearchSelect";
import { ArrowLeft } from "lucide-react";
import { extractProxyErrorMessage } from "@/lib/http/client";
import {
  aliasAdditionsFromRows,
  buildDiscoveredRows,
  buildModelCreationPayload,
  mergeModelGroupAliases,
  rowsPendingCreation,
  type CreationResult,
  type DiscoveredModelRow,
  type ModelGroupAliasMap,
} from "./wizardLogic";
import ReviewModelsStep from "./ReviewModelsStep";
import { DiscoverStep, JwksStep, ProviderStep, ResultsStep } from "./WizardSteps";

type WizardStep = "provider" | "credential" | "jwks" | "discover" | "review" | "creating" | "done";

const STEP_ORDER: readonly WizardStep[] = ["provider", "credential", "jwks", "discover", "review", "creating", "done"];

const STEP_LABELS: Record<WizardStep, string> = {
  provider: "Provider",
  credential: "Authentication",
  jwks: "Register issuer",
  discover: "Discover models",
  review: "Review models",
  creating: "Creating",
  done: "Done",
};

const ANTHROPIC_INTERNAL_ISSUER_DISCRIMINATOR = "internal_issuer";

const StepIndicator: React.FC<{ step: WizardStep; skipJwks: boolean }> = ({ step, skipJwks }) => {
  const visibleSteps = STEP_ORDER.filter((s) => s !== "creating" && (!skipJwks || s !== "jwks"));
  const currentIndex = visibleSteps.indexOf(step === "creating" ? "done" : step);
  return (
    <div className="mb-6 flex flex-wrap items-center gap-2 text-sm">
      {visibleSteps.map((s, index) => (
        <React.Fragment key={s}>
          {index > 0 && <span className="text-muted-foreground">{"->"}</span>}
          <span className={index <= currentIndex ? "font-medium text-foreground" : "text-muted-foreground"}>
            {STEP_LABELS[s]}
          </span>
        </React.Fragment>
      ))}
    </div>
  );
};

export default function AddProviderPanel() {
  const { accessToken } = useAuthorized();
  const queryClient = useQueryClient();
  const { data: providerMetadata } = useProviderFields();
  const { data: credentialsResponse } = useCredentials();

  const [step, setStep] = React.useState<WizardStep>("provider");
  const [selectedProvider, setSelectedProvider] = React.useState<Providers | null>(null);
  const [credentialName, setCredentialName] = React.useState("");
  const [savedCredential, setSavedCredential] = React.useState<{ name: string; provider: string } | null>(null);
  const [savedValues, setSavedValues] = React.useState<Record<string, unknown>>({});
  const [federationRuleId, setFederationRuleId] = React.useState("");
  const [jwks, setJwks] = React.useState<AnthropicJwks | null>(null);
  const [jwksError, setJwksError] = React.useState<string | null>(null);
  const [discoveryError, setDiscoveryError] = React.useState<string | null>(null);
  const [createError, setCreateError] = React.useState<string | null>(null);
  const [isDiscovering, setIsDiscovering] = React.useState(false);
  const [rows, setRows] = React.useState<DiscoveredModelRow[]>([]);
  const [isCreating, setIsCreating] = React.useState(false);
  const [creationResults, setCreationResults] = React.useState<CreationResult[]>([]);
  const [aliasCollisions, setAliasCollisions] = React.useState<string[]>([]);

  const form = useForm<MountedFormValues>({ mode: "onChange" });
  const registry = useMountRegistry();

  const providerOptions: SearchSelectOption[] = React.useMemo(
    () =>
      (providerMetadata ?? [])
        .slice()
        .sort((a, b) => a.provider_display_name.localeCompare(b.provider_display_name))
        .map((p) => ({
          label: p.provider_display_name,
          value: p.provider_display_name,
          icon: <ProviderLogo provider={p.provider_display_name} className="w-5 h-5" />,
        })),
    [providerMetadata],
  );

  const selectedProviderInfo: ProviderCreateInfo | undefined = React.useMemo(
    () => providerMetadata?.find((p) => p.provider_display_name === selectedProvider),
    [providerMetadata, selectedProvider],
  );
  const litellmProvider = selectedProviderInfo?.litellm_provider ?? "";

  // A credential is identified by name AND provider: switching provider under the same name must
  // create a new one, not PATCH the previous provider's credential into a different provider.
  const credentialSaved =
    savedCredential !== null && savedCredential.name === credentialName && savedCredential.provider === litellmProvider;

  const nameCollision =
    credentialName.length > 0 &&
    !credentialSaved &&
    (credentialsResponse?.credentials ?? []).some((c) => c.credential_name === credentialName);

  const goTo = (next: WizardStep) => setStep(next);

  const saveCredential = async () => {
    if (!accessToken || !selectedProvider) {
      return;
    }
    const isValid = await form.trigger(registry.mountedNames() as string[]);
    if (!isValid) {
      return;
    }
    const values = projectMountedValues(registry, form.getValues) as Record<string, unknown>;
    const nonEmptyValues = Object.fromEntries(
      Object.entries(values).filter(([, v]) => v !== "" && v !== undefined && v !== null),
    );
    try {
      if (!credentialSaved) {
        await credentialCreateCall(accessToken, {
          credential_name: credentialName,
          credential_values: nonEmptyValues,
          credential_info: { custom_llm_provider: litellmProvider },
        });
      } else {
        const credentialValuesToDelete = computeCredentialValuesToDelete(savedValues, values);
        const updatePayload = {
          credential_name: credentialName,
          credential_values: nonEmptyValues,
          credential_info: { custom_llm_provider: litellmProvider },
          ...(credentialValuesToDelete.length > 0 ? { credential_values_to_delete: credentialValuesToDelete } : {}),
        };
        await credentialUpdateCall(accessToken, credentialName, updatePayload);
      }
      setSavedValues(values);
      setSavedCredential({ name: credentialName, provider: litellmProvider });
      setFederationRuleId(
        typeof values.anthropic_federation_rule_id === "string" ? values.anthropic_federation_rule_id : "",
      );
      queryClient.invalidateQueries({ queryKey: ["credentials"] });
      toast.success(`Credential "${credentialName}" saved`);
      if (values.anthropic_identity_source === ANTHROPIC_INTERNAL_ISSUER_DISCRIMINATOR) {
        goTo("jwks");
        void loadJwks();
      } else {
        goTo("discover");
        void runDiscovery();
      }
    } catch (error) {
      toast.fromError(`Failed to save credential: ${extractProxyErrorMessage(error)}`);
    }
  };

  const loadJwks = async () => {
    if (!accessToken) return;
    setJwksError(null);
    try {
      const result = await getCredentialJwksCall(accessToken, credentialName);
      setJwks(result);
    } catch (error) {
      setJwksError(extractProxyErrorMessage(error));
    }
  };

  const confirmFederationRuleId = async () => {
    if (!accessToken) return;
    if (federationRuleId !== savedValues.anthropic_federation_rule_id) {
      try {
        await credentialUpdateCall(accessToken, credentialName, {
          credential_name: credentialName,
          credential_values: { anthropic_federation_rule_id: federationRuleId },
          credential_info: { custom_llm_provider: litellmProvider },
        });
        setSavedValues((prev) => ({ ...prev, anthropic_federation_rule_id: federationRuleId }));
      } catch (error) {
        toast.fromError(`Failed to save the federation rule id: ${extractProxyErrorMessage(error)}`);
        return;
      }
    }
    goTo("discover");
    void runDiscovery();
  };

  const runDiscovery = async () => {
    if (!accessToken) return;
    setIsDiscovering(true);
    setDiscoveryError(null);
    try {
      const result = await discoverProviderModelsCall(accessToken, {
        custom_llm_provider: litellmProvider,
        litellm_credential_name: credentialName,
      });
      setRows(buildDiscoveredRows(result.models));
      goTo("review");
    } catch (error) {
      setDiscoveryError(extractProxyErrorMessage(error));
    } finally {
      setIsDiscovering(false);
    }
  };

  const createModels = async () => {
    if (!accessToken) return;
    setIsCreating(true);
    setCreationResults([]);
    setAliasCollisions([]);
    setCreateError(null);
    goTo("creating");

    let existing: DeploymentInfoRow[];
    try {
      existing = (await listAllModelsCall(accessToken)).data;
    } catch (error) {
      setIsCreating(false);
      setCreateError(
        `Could not read the existing deployments, so creating now could duplicate ones already saved. ${extractProxyErrorMessage(error)}`,
      );
      goTo("review");
      return;
    }
    const pending = rowsPendingCreation(rows, litellmProvider, credentialName, existing);
    const pendingIds = new Set(pending.map((r) => r.id));

    const results: CreationResult[] = [];
    for (const row of rows) {
      if (!pendingIds.has(row.id)) {
        results.push({ row, status: "skipped", detail: "already created" });
        continue;
      }
      try {
        await createProviderModelCall(accessToken, buildModelCreationPayload(litellmProvider, credentialName, row));
        results.push({ row, status: "created" });
      } catch (error) {
        results.push({ row, status: "failed", detail: extractProxyErrorMessage(error) });
      }
    }
    setCreationResults(results);

    const failedRowIds = new Set(results.filter((r) => r.status === "failed").map((r) => r.row.id));
    const additions = aliasAdditionsFromRows(rows.filter((row) => !failedRowIds.has(row.id)));
    if (additions.length > 0) {
      try {
        const config = await getCallbacksCall(accessToken, "", "");
        const existingAliasMap: ModelGroupAliasMap = config?.router_settings?.model_group_alias ?? {};
        const { merged, collisions } = mergeModelGroupAliases(existingAliasMap, additions);
        if (collisions.length > 0) {
          setAliasCollisions([...collisions]);
        }
        await setCallbacksCall(accessToken, { router_settings: { model_group_alias: merged } });
      } catch (error) {
        toast.fromError(`Failed to save alternate names: ${extractProxyErrorMessage(error)}`);
      }
    }

    queryClient.invalidateQueries({ queryKey: ["models", "list"] });
    setIsCreating(false);
    goTo("done");
  };

  const isInternalIssuer = savedValues.anthropic_identity_source === ANTHROPIC_INTERNAL_ISSUER_DISCRIMINATOR;

  return (
    <div className="mx-auto max-w-3xl">
      <h2 className="mb-4 text-2xl font-semibold text-foreground">Add Provider</h2>
      <StepIndicator step={step} skipJwks={!isInternalIssuer} />

      {step === "provider" && (
        <ProviderStep
          providerOptions={providerOptions}
          selectedProvider={selectedProvider}
          onSelectProvider={(provider) => {
            if (provider !== selectedProvider) form.reset();
            setSelectedProvider(provider);
          }}
          credentialName={credentialName}
          onCredentialNameChange={setCredentialName}
          nameCollision={Boolean(nameCollision)}
          onNext={() => goTo("credential")}
        />
      )}

      {step === "credential" && selectedProvider && (
        <Card>
          <CardContent>
            <FormProvider {...form}>
              <MountedFormProvider value={{ control: form.control, registry }}>
                <form
                  onSubmit={(e) => {
                    e.preventDefault();
                    void saveCredential();
                  }}
                >
                  <ProviderSpecificFields selectedProvider={selectedProvider} />
                  <div className="flex justify-between">
                    <Button type="button" variant="outline" onClick={() => goTo("provider")}>
                      <ArrowLeft className="mr-1 size-4" /> Back
                    </Button>
                    <Button type="submit">{credentialSaved ? "Save changes" : "Save credential"}</Button>
                  </div>
                </form>
              </MountedFormProvider>
            </FormProvider>
          </CardContent>
        </Card>
      )}

      {step === "jwks" && (
        <JwksStep
          jwks={jwks}
          jwksError={jwksError}
          federationRuleId={federationRuleId}
          onFederationRuleIdChange={setFederationRuleId}
          onBack={() => goTo("credential")}
          onNext={() => void confirmFederationRuleId()}
        />
      )}

      {step === "discover" && (
        <DiscoverStep
          isDiscovering={isDiscovering}
          discoveryError={discoveryError}
          onBack={() => goTo(isInternalIssuer ? "jwks" : "credential")}
          onRetry={() => void runDiscovery()}
        />
      )}

      {step === "review" && (
        <ReviewModelsStep
          rows={rows}
          setRows={setRows}
          createError={createError}
          onBack={() => {
            goTo("discover");
            void runDiscovery();
          }}
          onCreateModels={() => void createModels()}
        />
      )}

      {(step === "creating" || step === "done") && (
        <ResultsStep
          isCreating={isCreating}
          isDone={step === "done"}
          creationResults={creationResults}
          aliasCollisions={aliasCollisions}
        />
      )}
    </div>
  );
}
