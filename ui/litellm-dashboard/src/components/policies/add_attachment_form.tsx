import React, { useCallback, useState, useEffect, useMemo } from "react";
import { Controller, FormProvider, useForm, useFormContext } from "react-hook-form";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { X } from "lucide-react";
import { Policy } from "./types";
import {
  teamListCall,
  keyListCall,
  modelAvailableCall,
  estimateAttachmentImpactCall,
} from "../networking";
import NotificationsManager from "../molecules/notifications_manager";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { buildAttachmentData } from "./build_attachment_data";
import ImpactPreviewAlert from "./impact_preview_alert";

interface AddAttachmentFormProps {
  visible: boolean;
  onClose: () => void;
  onSuccess: () => void;
  accessToken: string | null;
  policies: Policy[];
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  createAttachment: (accessToken: string, attachmentData: any) => Promise<any>;
}

interface AttachmentFormValues {
  policy_names: string[];
  teams: string[];
  keys: string[];
  models: string[];
  tags: string[];
}

function MultiSelect({
  value,
  onChange,
  options,
  placeholder,
  emptyText,
}: {
  value: string[];
  onChange: (next: string[]) => void;
  options: { label: string; value: string }[];
  placeholder: string;
  emptyText: string;
}) {
  const selected = useMemo(() => value ?? [], [value]);
  const remaining = useMemo(
    () => options.filter((o) => !selected.includes(o.value)),
    [options, selected],
  );

  return (
    <div className="space-y-2">
      <Select
        value=""
        onValueChange={(v) => {
          if (v) onChange([...selected, v]);
        }}
      >
        <SelectTrigger>
          <SelectValue placeholder={placeholder} />
        </SelectTrigger>
        <SelectContent>
          {remaining.length === 0 ? (
            <div className="py-2 px-3 text-sm text-muted-foreground">
              {emptyText}
            </div>
          ) : (
            remaining.map((opt) => (
              <SelectItem key={opt.value} value={opt.value}>
                {opt.label}
              </SelectItem>
            ))
          )}
        </SelectContent>
      </Select>
      {selected.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {selected.map((v) => {
            const opt = options.find((o) => o.value === v);
            return (
              <Badge
                key={v}
                variant="secondary"
                className="flex items-center gap-1"
              >
                {opt?.label ?? v}
                <button
                  type="button"
                  onClick={() => onChange(selected.filter((s) => s !== v))}
                  className="inline-flex items-center justify-center rounded-full hover:bg-muted-foreground/20"
                  aria-label={`Remove ${opt?.label ?? v}`}
                >
                  <X size={12} />
                </button>
              </Badge>
            );
          })}
        </div>
      )}
    </div>
  );
}

/**
 * Tag input that also supports selecting from a known list of suggestions
 * (teams, keys, models). Mirrors antd's Select mode="tags" behaviour.
 */
function SuggestibleTagInput({
  value,
  onChange,
  options,
  placeholder,
  loadingText,
  loading,
}: {
  value: string[];
  onChange: (next: string[]) => void;
  options: { label: string; value: string }[];
  placeholder: string;
  loadingText?: string;
  loading?: boolean;
}) {
  const [draft, setDraft] = useState("");
  const selected = value ?? [];
  const remaining = useMemo(
    () => options.filter((o) => !selected.includes(o.value)),
    [options, selected],
  );

  const commit = (raw: string) => {
    const parts = raw
      .split(",")
      .map((p) => p.trim())
      .filter(Boolean);
    const next = [...selected];
    for (const p of parts) {
      if (!next.includes(p)) next.push(p);
    }
    onChange(next);
    setDraft("");
  };

  return (
    <div className="space-y-2">
      <div className="flex items-start gap-2">
        <Input
          className="flex-1"
          value={draft}
          placeholder={loading ? loadingText || placeholder : placeholder}
          disabled={loading}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" || e.key === ",") {
              e.preventDefault();
              if (draft.trim()) commit(draft);
            } else if (e.key === "Backspace" && !draft && selected.length > 0) {
              onChange(selected.slice(0, -1));
            }
          }}
          onBlur={() => {
            if (draft.trim()) commit(draft);
          }}
        />
        {options.length > 0 && (
          <Select
            value=""
            onValueChange={(v) => {
              if (v && !selected.includes(v)) onChange([...selected, v]);
            }}
          >
            <SelectTrigger className="w-[180px]">
              <SelectValue placeholder="Pick…" />
            </SelectTrigger>
            <SelectContent>
              {remaining.length === 0 ? (
                <div className="py-2 px-3 text-sm text-muted-foreground">
                  No suggestions
                </div>
              ) : (
                remaining.map((opt) => (
                  <SelectItem key={opt.value} value={opt.value}>
                    {opt.label}
                  </SelectItem>
                ))
              )}
            </SelectContent>
          </Select>
        )}
      </div>
      {selected.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {selected.map((v) => (
            <Badge
              key={v}
              variant="secondary"
              className="flex items-center gap-1"
            >
              {v}
              <button
                type="button"
                onClick={() => onChange(selected.filter((s) => s !== v))}
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

interface FieldsProps {
  policyOptions: { label: string; value: string }[];
  scopeType: "global" | "specific";
  onScopeTypeChange: (scope: "global" | "specific") => void;
  availableTeams: string[];
  availableKeys: string[];
  availableModels: string[];
  isLoadingTeams: boolean;
  isLoadingKeys: boolean;
  isLoadingModels: boolean;
}

function AttachmentFormFields({
  policyOptions,
  scopeType,
  onScopeTypeChange,
  availableTeams,
  availableKeys,
  availableModels,
  isLoadingTeams,
  isLoadingKeys,
  isLoadingModels,
}: FieldsProps) {
  const { control, formState } = useFormContext<AttachmentFormValues>();

  return (
    <>
      <div className="space-y-2">
        <Label>
          Policies <span className="text-destructive">*</span>
        </Label>
        <Controller
          control={control}
          name="policy_names"
          rules={{
            validate: (v) =>
              (v && v.length > 0) || "Please select at least one policy",
          }}
          render={({ field }) => (
            <MultiSelect
              value={field.value ?? []}
              onChange={field.onChange}
              options={policyOptions}
              placeholder="Select policies to attach"
              emptyText="No policies available"
            />
          )}
        />
        {formState.errors.policy_names && (
          <p className="text-sm text-destructive">
            {formState.errors.policy_names.message as string}
          </p>
        )}
      </div>

      <div className="flex items-center gap-2 mt-4 mb-2">
        <span className="font-bold">Scope</span>
        <hr className="flex-1 border-border" />
      </div>

      <div className="space-y-2">
        <Label>Scope Type</Label>
        <RadioGroup
          value={scopeType}
          onValueChange={(v) => onScopeTypeChange(v as "global" | "specific")}
          className="flex flex-col gap-2"
        >
          <label className="flex items-center gap-2 cursor-pointer">
            <RadioGroupItem value="specific" />
            Specific (teams, keys, models, or tags)
          </label>
          <label className="flex items-center gap-2 cursor-pointer">
            <RadioGroupItem value="global" />
            Global (applies to all requests)
          </label>
        </RadioGroup>
      </div>

      {scopeType === "specific" && (
        <>
          <div className="space-y-2 mt-4">
            <Label>Teams</Label>
            <p className="text-xs text-muted-foreground">
              Select team aliases or enter custom patterns. Supports wildcards
              (e.g., healthcare-*).
            </p>
            <Controller
              control={control}
              name="teams"
              render={({ field }) => (
                <SuggestibleTagInput
                  value={field.value ?? []}
                  onChange={field.onChange}
                  options={availableTeams.map((t) => ({ label: t, value: t }))}
                  placeholder="Enter team alias and press Enter"
                  loadingText="Loading teams..."
                  loading={isLoadingTeams}
                />
              )}
            />
          </div>

          <div className="space-y-2 mt-4">
            <Label>Keys</Label>
            <p className="text-xs text-muted-foreground">
              Select key aliases or enter custom patterns. Supports wildcards
              (e.g., dev-*).
            </p>
            <Controller
              control={control}
              name="keys"
              render={({ field }) => (
                <SuggestibleTagInput
                  value={field.value ?? []}
                  onChange={field.onChange}
                  options={availableKeys.map((k) => ({ label: k, value: k }))}
                  placeholder="Enter key alias and press Enter"
                  loadingText="Loading keys..."
                  loading={isLoadingKeys}
                />
              )}
            />
          </div>

          <div className="space-y-2 mt-4">
            <Label>Models</Label>
            <p className="text-xs text-muted-foreground">
              Model names this attachment applies to. Supports wildcards (e.g.,
              gpt-4*). Leave empty to apply to all models.
            </p>
            <Controller
              control={control}
              name="models"
              render={({ field }) => (
                <SuggestibleTagInput
                  value={field.value ?? []}
                  onChange={field.onChange}
                  options={availableModels.map((m) => ({ label: m, value: m }))}
                  placeholder="Enter model name and press Enter"
                  loadingText="Loading models..."
                  loading={isLoadingModels}
                />
              )}
            />
          </div>

          <div className="space-y-2 mt-4">
            <Label>Tags</Label>
            <p className="text-xs text-muted-foreground">
              Matches tags from key/team <code>metadata.tags</code> or tags
              passed dynamically in the request body. Use <code>*</code> as a
              suffix wildcard (e.g., <code>prod-*</code> matches{" "}
              <code>prod-us</code>, <code>prod-eu</code>).
            </p>
            <Controller
              control={control}
              name="tags"
              render={({ field }) => (
                <SuggestibleTagInput
                  value={field.value ?? []}
                  onChange={field.onChange}
                  options={[]}
                  placeholder="Type a tag and press Enter (e.g. healthcare, prod-*)"
                />
              )}
            />
          </div>
        </>
      )}
    </>
  );
}

const AddAttachmentForm: React.FC<AddAttachmentFormProps> = ({
  visible,
  onClose,
  onSuccess,
  accessToken,
  policies,
  createAttachment,
}) => {
  const form = useForm<AttachmentFormValues>({
    defaultValues: {
      policy_names: [],
      teams: [],
      keys: [],
      models: [],
      tags: [],
    },
  });
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [scopeType, setScopeType] = useState<"global" | "specific">("global");
  const [availableTeams, setAvailableTeams] = useState<string[]>([]);
  const [availableKeys, setAvailableKeys] = useState<string[]>([]);
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [isLoadingTeams, setIsLoadingTeams] = useState(false);
  const [isLoadingKeys, setIsLoadingKeys] = useState(false);
  const [isLoadingModels, setIsLoadingModels] = useState(false);
  const [isEstimating, setIsEstimating] = useState(false);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const [impactResult, setImpactResult] = useState<any>(null);
  const { userId, userRole } = useAuthorized();

  const loadTeamsKeysAndModels = useCallback(async () => {
    if (!accessToken) return;

    setIsLoadingTeams(true);
    try {
      const teamsResponse = await teamListCall(accessToken, null, userId);
      const teamsArray = Array.isArray(teamsResponse)
        ? teamsResponse
        : teamsResponse?.data || [];
      const teamAliases = teamsArray
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        .map((t: any) => t.team_alias)
        .filter(Boolean);
      setAvailableTeams(teamAliases);
    } catch (error) {
      console.error("Failed to load teams:", error);
    } finally {
      setIsLoadingTeams(false);
    }

    setIsLoadingKeys(true);
    try {
      const keysResponse = await keyListCall(
        accessToken,
        null,
        null,
        null,
        null,
        null,
        1,
        100,
      );
      const keysArray = keysResponse?.keys || keysResponse?.data || [];
      const keyAliases = keysArray
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        .map((k: any) => k.key_alias)
        .filter(Boolean);
      setAvailableKeys(keyAliases);
    } catch (error) {
      console.error("Failed to load keys:", error);
    } finally {
      setIsLoadingKeys(false);
    }

    setIsLoadingModels(true);
    try {
      const modelsResponse = await modelAvailableCall(
        accessToken,
        userId || "",
        userRole || "",
      );
      const modelsArray =
        modelsResponse?.data ||
        (Array.isArray(modelsResponse) ? modelsResponse : []);
      const modelIds = modelsArray
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        .map((m: any) => m.id || m.model_name)
        .filter(Boolean);
      setAvailableModels(modelIds);
    } catch (error) {
      console.error("Failed to load models:", error);
    } finally {
      setIsLoadingModels(false);
    }
  }, [accessToken, userId, userRole]);

  useEffect(() => {
    if (visible && accessToken) {
      loadTeamsKeysAndModels();
    }
  }, [visible, accessToken, loadTeamsKeysAndModels]);

  const resetForm = () => {
    form.reset({
      policy_names: [],
      teams: [],
      keys: [],
      models: [],
      tags: [],
    });
    setScopeType("global");
    setImpactResult(null);
  };

  const handlePreviewImpact = async () => {
    if (!accessToken) return;
    const valid = await form.trigger("policy_names");
    if (!valid) return;
    setIsEstimating(true);
    try {
      const values = form.getValues();
      const firstPolicy = values.policy_names?.[0];
      if (!firstPolicy) return;
      const data = buildAttachmentData(
        {
          ...values,
          policy_name: firstPolicy,
        },
        scopeType,
      );
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

  const onSubmit = form.handleSubmit(async (values) => {
    try {
      setIsSubmitting(true);

      if (!accessToken) {
        throw new Error("No access token available");
      }

      const selectedPolicyNames: string[] = values.policy_names || [];

      const results = await Promise.allSettled(
        selectedPolicyNames.map((policyName) => {
          const data = buildAttachmentData(
            {
              ...values,
              policy_name: policyName,
            },
            scopeType,
          );
          return createAttachment(accessToken, data);
        }),
      );

      const successCount = results.filter(
        (r) => r.status === "fulfilled",
      ).length;
      const failed = results.filter(
        (r) => r.status === "rejected",
      ) as PromiseRejectedResult[];

      if (successCount > 0 && failed.length === 0) {
        NotificationsManager.success(
          successCount === 1
            ? "Attachment created successfully"
            : `${successCount} attachments created successfully`,
        );
      } else if (successCount > 0 && failed.length > 0) {
        NotificationsManager.fromBackend(
          `${successCount} attachments created, ${failed.length} failed`,
        );
      } else {
        throw new Error(
          failed[0]?.reason instanceof Error
            ? failed[0].reason.message
            : "Failed to create attachments",
        );
      }

      resetForm();
      onSuccess();
      onClose();
    } catch (error) {
      console.error("Failed to create attachment:", error);
      NotificationsManager.fromBackend(
        "Failed to create attachment: " +
          (error instanceof Error ? error.message : String(error)),
      );
    } finally {
      setIsSubmitting(false);
    }
  });

  const policyOptions = policies.map((p) => ({
    label: p.policy_name,
    value: p.policy_name,
  }));

  return (
    <Dialog
      open={visible}
      onOpenChange={(o) => (!o ? handleClose() : undefined)}
    >
      <DialogContent className="max-w-[600px] max-h-[85vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>Create Policy Attachment</DialogTitle>
        </DialogHeader>
        <FormProvider {...form}>
          <form onSubmit={onSubmit} className="space-y-2">
            <AttachmentFormFields
              policyOptions={policyOptions}
              scopeType={scopeType}
              onScopeTypeChange={setScopeType}
              availableTeams={availableTeams}
              availableKeys={availableKeys}
              availableModels={availableModels}
              isLoadingTeams={isLoadingTeams}
              isLoadingKeys={isLoadingKeys}
              isLoadingModels={isLoadingModels}
            />

            {impactResult && <ImpactPreviewAlert impactResult={impactResult} />}

            <div className="flex justify-end space-x-2 mt-4">
              <Button type="button" variant="outline" onClick={handleClose}>
                Cancel
              </Button>
              {scopeType === "specific" && (
                <Button
                  type="button"
                  variant="secondary"
                  onClick={handlePreviewImpact}
                  disabled={isEstimating}
                >
                  {isEstimating ? "Estimating..." : "Estimate Impact"}
                </Button>
              )}
              <Button type="submit" disabled={isSubmitting}>
                {isSubmitting ? "Creating..." : "Create Attachment"}
              </Button>
            </div>
          </form>
        </FormProvider>
      </DialogContent>
    </Dialog>
  );
};

export default AddAttachmentForm;
