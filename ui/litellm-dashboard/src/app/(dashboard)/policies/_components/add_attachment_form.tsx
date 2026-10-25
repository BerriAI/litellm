import { useLanguage } from "@/contexts/LanguageContext";
import React, { useState, useEffect, useMemo } from "react";
import { englishTranslate, type Translate } from "@/i18n/language";
import { CircleHelp } from "lucide-react";
import { z } from "zod";
import { Policy } from "@/components/policies/types";
import { teamListCall, keyListCall, modelAvailableCall, estimateAttachmentImpactCall } from "@/components/networking";
import { toast } from "@/lib/toast";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { FieldGroup, FieldLabel, FieldTitle } from "@/components/ui/field";
import { FormField } from "@/components/shared/form/FormField";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Separator } from "@/components/ui/separator";
import { Switch } from "@/components/ui/switch";
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
  priority: number | null;
  default: boolean;
}

const EMPTY_VALUES: AttachmentFormValues = {
  policy_names: [],
  teams: [],
  keys: [],
  models: [],
  tags: [],
  priority: null,
  default: false,
};

const INT32_MIN = -2147483648;
const INT32_MAX = 2147483647;

const attachmentShape = (t: Translate) => ({
  policy_names: z.array(z.string()).min(1, t("Please select at least one policy")),
  teams: z.array(z.string()),
  keys: z.array(z.string()),
  models: z.array(z.string()),
  tags: z.array(z.string()),
  priority: z
    .number({ error: t("Priority must be a whole number") })
    .int(t("Priority must be a whole number"))
    .min(INT32_MIN, t("Priority must be at least {minimum}", { minimum: INT32_MIN }))
    .max(INT32_MAX, t("Priority must be at most {maximum}", { maximum: INT32_MAX }))
    .nullable(),
  default: z.boolean(),
});

// 只重建校验器，不重置正在编辑的策略绑定。
const buildAttachmentSchema = (
  scopeType: ScopeType,
  teamsLoaded: boolean,
  availableTeams: string[],
  t: Translate = englishTranslate,
) =>
  z.object(attachmentShape(t)).superRefine((values, ctx) => {
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
      message: t(
        'These teams don\'t exist: {teams}. Choose an existing team, or use a wildcard like "team-*" to match by prefix.',
        { teams: invalid.join(", ") },
      ),
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
  const { t } = useLanguage();
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
  const schema = useMemo(
    () => buildAttachmentSchema(scopeType, teamsLoaded, availableTeams, t),
    [scopeType, teamsLoaded, availableTeams, t],
  );
  const form = useZodForm(schema, {
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
          successCount === 1
            ? t("Attachment created successfully")
            : t("{count} attachments created successfully", { count: successCount }),
        );
      } else if (successCount > 0 && failed.length > 0) {
        toast.fromError(
          t("{successCount} attachments created, {count} failed", {
            successCount: String(successCount),
            count: String(failed.length),
          }),
        );
      } else {
        throw new Error(
          failed[0]?.reason instanceof Error ? failed[0].reason.message : t("Failed to create attachments"),
        );
      }

      resetForm();
      onSuccess();
      onClose();
    } catch (error) {
      console.error("Failed to create attachment:", error);
      toast.fromError(
        t("Failed to create attachment: {error}", { error: error instanceof Error ? error.message : String(error) }),
      );
    } finally {
      setIsSubmitting(false);
    }
  };

  const policyOptions = policies.map((p) => p.policy_name);

  return (
    <Dialog open={visible} onOpenChange={(open) => !open && handleClose()}>
      <DialogContent className="max-h-[calc(100dvh-2rem)] overflow-y-auto sm:max-w-[600px]">
        <DialogHeader>
          <DialogTitle>{t("Create Policy Attachment")}</DialogTitle>
        </DialogHeader>
        <TooltipProvider>
          <form onSubmit={(event) => event.preventDefault()} noValidate>
            <FieldGroup>
              <FormField control={form.control} name="policy_names" label={t("Policies")}>
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
                    placeholder={t("Select policies to attach")}
                    options={policyOptions}
                    emptyText={t("No matching policies")}
                    ariaInvalid={ariaInvalid}
                    ariaDescribedBy={ariaDescribedBy}
                  />
                )}
              </FormField>

              <div className="flex items-center gap-3">
                <span className="text-sm font-semibold">{t("Scope")}</span>
                <Separator className="flex-1" />
              </div>

              <div>
                <FieldTitle className="mb-2">{t("Scope Type")}</FieldTitle>
                <RadioGroup value={scopeType} onValueChange={(value: unknown) => setScopeType(value as ScopeType)}>
                  <FieldLabel className="font-normal">
                    <RadioGroupItem value="specific" />
                    {t("Specific (teams, keys, models, or tags)")}
                  </FieldLabel>
                  <FieldLabel className="font-normal">
                    <RadioGroupItem value="global" />
                    {t("Global (applies to all requests)")}
                  </FieldLabel>
                </RadioGroup>
              </div>

              {scopeType === "specific" && (
                <>
                  <FormField
                    control={form.control}
                    name="teams"
                    label={labelWithHint(
                      t("Teams"),
                      t("Select team aliases or enter custom patterns. Supports wildcards (e.g., healthcare-*)"),
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
                        placeholder={isLoadingTeams ? t("Loading teams...") : t("Select or enter team aliases")}
                        options={availableTeams}
                        allowCustomValues
                        tokenSeparators={[","]}
                        emptyText={t("No matching teams")}
                        ariaInvalid={ariaInvalid}
                        ariaDescribedBy={ariaDescribedBy}
                      />
                    )}
                  </FormField>

                  <FormField
                    control={form.control}
                    name="keys"
                    label={labelWithHint(
                      t("Keys"),
                      t("Select key aliases or enter custom patterns. Supports wildcards (e.g., dev-*)"),
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
                        placeholder={isLoadingKeys ? t("Loading keys...") : t("Select or enter key aliases")}
                        options={availableKeys}
                        allowCustomValues
                        tokenSeparators={[","]}
                        emptyText={t("No matching keys")}
                        ariaInvalid={ariaInvalid}
                        ariaDescribedBy={ariaDescribedBy}
                      />
                    )}
                  </FormField>

                  <FormField
                    control={form.control}
                    name="models"
                    label={labelWithHint(
                      t("Models"),
                      t(
                        "Model names this attachment applies to. Supports wildcards (e.g., gpt-4*). Leave empty to apply to all models.",
                      ),
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
                          isLoadingModels
                            ? t("Loading models...")
                            : t("Select or enter model names (e.g., gpt-4, bedrock/*)")
                        }
                        options={availableModels}
                        allowCustomValues
                        tokenSeparators={[","]}
                        emptyText={t("No matching models")}
                        ariaInvalid={ariaInvalid}
                        ariaDescribedBy={ariaDescribedBy}
                      />
                    )}
                  </FormField>

                  <FormField
                    control={form.control}
                    name="tags"
                    label={labelWithHint(
                      t("Tags"),
                      t(
                        "Match against tags set in key or team metadata. Use exact values (e.g., healthcare) or wildcard patterns (e.g., health-*) where * matches any suffix.",
                      ),
                    )}
                    description={
                      <span className="text-xs">
                        {t("Matches tags from key/team")} <code>metadata.tags</code>{" "}
                        {t("or tags passed dynamically in the request body. Use")} <code>*</code>{" "}
                        {t("as a suffix wildcard (e.g.,")} <code>prod-*</code> {t("matches")} <code>prod-us</code>,{" "}
                        <code>prod-eu</code>).
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
                        placeholder={t("Type a tag and press Enter (e.g. healthcare, prod-*)")}
                        allowCustomValues
                        tokenSeparators={[",", " "]}
                        ariaInvalid={ariaInvalid}
                        ariaDescribedBy={ariaDescribedBy}
                      />
                    )}
                  </FormField>
                </>
              )}

              <FormField
                control={form.control}
                name="priority"
                label={labelWithHint(
                  t("Priority"),
                  t("Lower numbers run first. Attachments with a priority run before attachments without one."),
                )}
                description={t(
                  "Optional. Leave blank to keep the default order: global, then teams, keys, tags, models.",
                )}
              >
                {({ ref, value, onChange, ...field }) => (
                  <Input
                    {...field}
                    ref={ref}
                    type="number"
                    step={1}
                    value={value ?? ""}
                    placeholder="e.g. 10"
                    onChange={(event) => onChange(event.target.value === "" ? null : event.target.valueAsNumber)}
                  />
                )}
              </FormField>

              <FormField
                control={form.control}
                name="default"
                label={labelWithHint(
                  t("Default (fallback)"),
                  t("A default attachment is applied only when no non-default attachment matches the request."),
                )}
                description={t("Use this for the guardrail everyone gets unless they opt in to another attachment.")}
              >
                {({ value, onChange, ref, ...field }) => (
                  <Switch {...field} inputRef={ref} checked={value === true} onCheckedChange={onChange} />
                )}
              </FormField>
            </FieldGroup>

            {impactResult && <ImpactPreviewAlert impactResult={impactResult} isDefault={form.watch("default")} />}

            <div className="flex justify-end space-x-2 mt-4">
              <Button type="button" variant="secondary" onClick={handleClose}>
                {t("Cancel")}
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
                  {t("Estimate Impact")}
                </Button>
              )}
              <Button
                type="button"
                onClick={form.handleSubmit(handleSubmit)}
                disabled={isSubmitting}
                aria-busy={isSubmitting}
              >
                {isSubmitting && <UiLoadingSpinner className="size-4" />}
                {t("Create Attachment")}
              </Button>
            </div>
          </form>
        </TooltipProvider>
      </DialogContent>
    </Dialog>
  );
};

export default AddAttachmentForm;
