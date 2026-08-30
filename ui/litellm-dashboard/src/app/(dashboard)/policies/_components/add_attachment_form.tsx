import React, { useState, useEffect } from "react";
import { CircleHelp } from "lucide-react";
import { z } from "zod/v4";
import { Policy } from "@/components/policies/types";
import { teamListCall, keyListCall, modelAvailableCall, estimateAttachmentImpactCall } from "@/components/networking";
import { toast } from "@/lib/toast";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { FieldGroup, FieldLabel, FieldTitle } from "@/components/ui/field";
import { FormField } from "@/components/shared/form/FormField";
import { Button } from "@/components/ui/button";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Separator } from "@/components/ui/separator";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { UiLoadingSpinner } from "@/components/ui/ui-loading-spinner";
import { useZodForm } from "@/lib/forms/useZodForm";
import { buildAttachmentData } from "./build_attachment_data";
import { getInvalidTeamEntries } from "./scope_validation";
import ImpactPreviewAlert from "./impact_preview_alert";
import { TokenSelect } from "./TokenSelect";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";

interface AddAttachmentFormProps {
  visible: boolean;
  onClose: () => void;
  onSuccess: () => void;
  accessToken: string | null;
  policies: Policy[];
  createAttachment: (accessToken: string, attachmentData: any) => Promise<any>;
}

type ScopeType = "global" | "specific";

interface AttachmentFormValues {
  policy_names: string[];
  teams: string[];
  keys: string[];
  models: string[];
  tags: string[];
}

const EMPTY_VALUES: AttachmentFormValues = {
  policy_names: [],
  teams: [],
  keys: [],
  models: [],
  tags: [],
};

const attachmentShape = {
  policy_names: z.array(z.string()).min(1, "Please select at least one policy"),
  teams: z.array(z.string()),
  keys: z.array(z.string()),
  models: z.array(z.string()),
  tags: z.array(z.string()),
};

const buildAttachmentSchema = (scopeType: ScopeType, teamsLoaded: boolean, availableTeams: string[]) =>
  z.object(attachmentShape).superRefine((values, ctx) => {
    if (scopeType !== "specific" || !teamsLoaded) {
      return;
    }
    const invalid = getInvalidTeamEntries(values.teams, availableTeams);
    if (invalid.length === 0) {
      return;
    }
    ctx.addIssue({
      code: "custom",
      path: ["teams"],
      message:
        `These teams don't exist: ${invalid.join(", ")}. ` +
        `Choose an existing team, or use a wildcard like "team-*" to match by prefix.`,
    });
  });

const labelWithHint = (label: string, hint: string): React.ReactNode => (
  <>
    {label}
    <Tooltip>
      <TooltipTrigger render={<CircleHelp className="size-3.5 shrink-0 cursor-help text-muted-foreground" />} />
      <TooltipContent>{hint}</TooltipContent>
    </Tooltip>
  </>
);

const AddAttachmentForm: React.FC<AddAttachmentFormProps> = ({
  visible,
  onClose,
  onSuccess,
  accessToken,
  policies,
  createAttachment,
}) => {
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [scopeType, setScopeType] = useState<ScopeType>("global");
  const [availableTeams, setAvailableTeams] = useState<string[]>([]);
  const [teamsLoaded, setTeamsLoaded] = useState(false);
  const [availableKeys, setAvailableKeys] = useState<string[]>([]);
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [isLoadingTeams, setIsLoadingTeams] = useState(false);
  const [isLoadingKeys, setIsLoadingKeys] = useState(false);
  const [isLoadingModels, setIsLoadingModels] = useState(false);
  const [isEstimating, setIsEstimating] = useState(false);
  const [impactResult, setImpactResult] = useState<any>(null);
  const { userId, userRole } = useAuthorized();
  const form = useZodForm(buildAttachmentSchema(scopeType, teamsLoaded, availableTeams), {
    defaultValues: EMPTY_VALUES,
  });

  useEffect(() => {
    if (visible && accessToken) {
      loadTeamsKeysAndModels();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [visible, accessToken]);

  const loadTeamsKeysAndModels = async () => {
    if (!accessToken) return;

    // Load teams — teamListCall returns a plain array of team objects
    setIsLoadingTeams(true);
    setTeamsLoaded(false);
    try {
      const teamsResponse = await teamListCall(accessToken, null, null);
      const teamsArray = Array.isArray(teamsResponse) ? teamsResponse : teamsResponse?.data || [];
      const teamAliases = teamsArray.map((t: any) => t.team_alias).filter(Boolean);
      setAvailableTeams(teamAliases);
      setTeamsLoaded(true);
    } catch (error) {
      console.error("Failed to load teams:", error);
    } finally {
      setIsLoadingTeams(false);
    }

    // Load keys — keyListCall returns {keys: [...], total_count, ...}
    setIsLoadingKeys(true);
    try {
      const keysResponse = await keyListCall(accessToken, null, null, null, null, null, 1, 100);
      const keysArray = keysResponse?.keys || keysResponse?.data || [];
      const keyAliases = keysArray.map((k: any) => k.key_alias).filter(Boolean);
      setAvailableKeys(keyAliases);
    } catch (error) {
      console.error("Failed to load keys:", error);
    } finally {
      setIsLoadingKeys(false);
    }

    // Load models
    setIsLoadingModels(true);
    try {
      const modelsResponse = await modelAvailableCall(accessToken, userId || "", userRole || "");
      const modelsArray = modelsResponse?.data || (Array.isArray(modelsResponse) ? modelsResponse : []);
      const modelIds = modelsArray.map((m: any) => m.id || m.model_name).filter(Boolean);
      setAvailableModels(modelIds);
    } catch (error) {
      console.error("Failed to load models:", error);
    } finally {
      setIsLoadingModels(false);
    }
  };

  const resetForm = () => {
    form.reset(EMPTY_VALUES);
    setScopeType("global");
    setImpactResult(null);
  };

  const handlePreviewImpact = async () => {
    if (!accessToken) return;
    if (!(await form.trigger("policy_names"))) {
      return;
    }
    setIsEstimating(true);
    try {
      const values = form.getValues();
      const firstPolicy = values.policy_names[0];
      if (!firstPolicy) return;
      const data = buildAttachmentData({ ...values, policy_name: firstPolicy }, scopeType);
      const result = await estimateAttachmentImpactCall(accessToken, data);
      setImpactResult(result);
    } catch (error) {
      console.error("Failed to estimate impact:", error);
    } finally {
      setIsEstimating(false);
    }
  };

  const handleClose = () => {
    resetForm();
    onClose();
  };

  const handleSubmit = async (values: AttachmentFormValues) => {
    try {
      setIsSubmitting(true);

      if (!accessToken) {
        throw new Error("No access token available");
      }

      const results = await Promise.allSettled(
        values.policy_names.map((policyName) => {
          const data = buildAttachmentData({ ...values, policy_name: policyName }, scopeType);
          return createAttachment(accessToken, data);
        }),
      );

      const successCount = results.filter((r) => r.status === "fulfilled").length;
      const failed = results.filter((r) => r.status === "rejected") as PromiseRejectedResult[];

      if (successCount > 0 && failed.length === 0) {
        toast.success(
          successCount === 1 ? "Attachment created successfully" : `${successCount} attachments created successfully`,
        );
      } else if (successCount > 0 && failed.length > 0) {
        toast.fromError(`${successCount} attachments created, ${failed.length} failed`);
      } else {
        throw new Error(failed[0]?.reason instanceof Error ? failed[0].reason.message : "Failed to create attachments");
      }

      resetForm();
      onSuccess();
      onClose();
    } catch (error) {
      console.error("Failed to create attachment:", error);
      toast.fromError("Failed to create attachment: " + (error instanceof Error ? error.message : String(error)));
    } finally {
      setIsSubmitting(false);
    }
  };

  const policyOptions = policies.map((p) => p.policy_name);

  return (
    <Dialog open={visible} onOpenChange={(open) => !open && handleClose()}>
      <DialogContent className="max-h-[calc(100dvh-2rem)] overflow-y-auto sm:max-w-[600px]">
        <DialogHeader>
          <DialogTitle>Create Policy Attachment</DialogTitle>
        </DialogHeader>
        <TooltipProvider>
          <form onSubmit={(event) => event.preventDefault()} noValidate>
            <FieldGroup>
              <FormField control={form.control} name="policy_names" label="Policies">
                {({
                  id,
                  value,
                  onChange,
                  onBlur,
                  "aria-invalid": ariaInvalid,
                  "aria-describedby": ariaDescribedBy,
                }) => (
                  <TokenSelect
                    id={id}
                    value={value}
                    onValueChange={onChange}
                    onBlur={onBlur}
                    placeholder="Select policies to attach"
                    options={policyOptions}
                    emptyText="No matching policies"
                    ariaInvalid={ariaInvalid}
                    ariaDescribedBy={ariaDescribedBy}
                  />
                )}
              </FormField>

              <div className="flex items-center gap-3">
                <span className="text-sm font-semibold">Scope</span>
                <Separator className="flex-1" />
              </div>

              <div>
                <FieldTitle className="mb-2">Scope Type</FieldTitle>
                <RadioGroup value={scopeType} onValueChange={(value: unknown) => setScopeType(value as ScopeType)}>
                  <FieldLabel className="font-normal">
                    <RadioGroupItem value="specific" />
                    Specific (teams, keys, models, or tags)
                  </FieldLabel>
                  <FieldLabel className="font-normal">
                    <RadioGroupItem value="global" />
                    Global (applies to all requests)
                  </FieldLabel>
                </RadioGroup>
              </div>

              {scopeType === "specific" && (
                <>
                  <FormField
                    control={form.control}
                    name="teams"
                    label={labelWithHint(
                      "Teams",
                      "Select team aliases or enter custom patterns. Supports wildcards (e.g., healthcare-*)",
                    )}
                  >
                    {({
                      id,
                      value,
                      onChange,
                      onBlur,
                      "aria-invalid": ariaInvalid,
                      "aria-describedby": ariaDescribedBy,
                    }) => (
                      <TokenSelect
                        id={id}
                        value={value}
                        onValueChange={onChange}
                        onBlur={onBlur}
                        placeholder={isLoadingTeams ? "Loading teams..." : "Select or enter team aliases"}
                        options={availableTeams}
                        allowCustomValues
                        tokenSeparators={[","]}
                        emptyText="No matching teams"
                        ariaInvalid={ariaInvalid}
                        ariaDescribedBy={ariaDescribedBy}
                      />
                    )}
                  </FormField>

                  <FormField
                    control={form.control}
                    name="keys"
                    label={labelWithHint(
                      "Keys",
                      "Select key aliases or enter custom patterns. Supports wildcards (e.g., dev-*)",
                    )}
                  >
                    {({
                      id,
                      value,
                      onChange,
                      onBlur,
                      "aria-invalid": ariaInvalid,
                      "aria-describedby": ariaDescribedBy,
                    }) => (
                      <TokenSelect
                        id={id}
                        value={value}
                        onValueChange={onChange}
                        onBlur={onBlur}
                        placeholder={isLoadingKeys ? "Loading keys..." : "Select or enter key aliases"}
                        options={availableKeys}
                        allowCustomValues
                        tokenSeparators={[","]}
                        emptyText="No matching keys"
                        ariaInvalid={ariaInvalid}
                        ariaDescribedBy={ariaDescribedBy}
                      />
                    )}
                  </FormField>

                  <FormField
                    control={form.control}
                    name="models"
                    label={labelWithHint(
                      "Models",
                      "Model names this attachment applies to. Supports wildcards (e.g., gpt-4*). Leave empty to apply to all models.",
                    )}
                  >
                    {({
                      id,
                      value,
                      onChange,
                      onBlur,
                      "aria-invalid": ariaInvalid,
                      "aria-describedby": ariaDescribedBy,
                    }) => (
                      <TokenSelect
                        id={id}
                        value={value}
                        onValueChange={onChange}
                        onBlur={onBlur}
                        placeholder={
                          isLoadingModels ? "Loading models..." : "Select or enter model names (e.g., gpt-4, bedrock/*)"
                        }
                        options={availableModels}
                        allowCustomValues
                        tokenSeparators={[","]}
                        emptyText="No matching models"
                        ariaInvalid={ariaInvalid}
                        ariaDescribedBy={ariaDescribedBy}
                      />
                    )}
                  </FormField>

                  <FormField
                    control={form.control}
                    name="tags"
                    label={labelWithHint(
                      "Tags",
                      "Match against tags set in key or team metadata. Use exact values (e.g., healthcare) or wildcard patterns (e.g., health-*) where * matches any suffix.",
                    )}
                    description={
                      <span className="text-xs">
                        Matches tags from key/team <code>metadata.tags</code> or tags passed dynamically in the request
                        body. Use <code>*</code> as a suffix wildcard (e.g., <code>prod-*</code> matches{" "}
                        <code>prod-us</code>, <code>prod-eu</code>).
                      </span>
                    }
                  >
                    {({
                      id,
                      value,
                      onChange,
                      onBlur,
                      "aria-invalid": ariaInvalid,
                      "aria-describedby": ariaDescribedBy,
                    }) => (
                      <TokenSelect
                        id={id}
                        value={value}
                        onValueChange={onChange}
                        onBlur={onBlur}
                        placeholder="Type a tag and press Enter (e.g. healthcare, prod-*)"
                        allowCustomValues
                        tokenSeparators={[",", " "]}
                        ariaInvalid={ariaInvalid}
                        ariaDescribedBy={ariaDescribedBy}
                      />
                    )}
                  </FormField>
                </>
              )}
            </FieldGroup>

            {impactResult && <ImpactPreviewAlert impactResult={impactResult} />}

            <div className="flex justify-end space-x-2 mt-4">
              <Button type="button" variant="secondary" onClick={handleClose}>
                Cancel
              </Button>
              {scopeType === "specific" && (
                <Button
                  type="button"
                  variant="secondary"
                  onClick={handlePreviewImpact}
                  disabled={isEstimating}
                  aria-busy={isEstimating}
                >
                  {isEstimating && <UiLoadingSpinner className="size-4" />}
                  Estimate Impact
                </Button>
              )}
              <Button
                type="button"
                onClick={form.handleSubmit(handleSubmit)}
                disabled={isSubmitting}
                aria-busy={isSubmitting}
              >
                {isSubmitting && <UiLoadingSpinner className="size-4" />}
                Create Attachment
              </Button>
            </div>
          </form>
        </TooltipProvider>
      </DialogContent>
    </Dialog>
  );
};

export default AddAttachmentForm;
