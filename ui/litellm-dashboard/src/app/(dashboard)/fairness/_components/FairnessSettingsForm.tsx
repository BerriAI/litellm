"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import * as React from "react";
import { useFieldArray, useWatch, type Control } from "react-hook-form";

import { WORKLOAD_CLASS_QUERY_KEY } from "@/components/fairness/WorkloadClassSelect";
import { FormField } from "@/components/shared/form/FormField";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { FieldGroup } from "@/components/ui/field";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import { Switch } from "@/components/ui/switch";
import { useZodForm } from "@/lib/forms/useZodForm";
import { fetchClient } from "@/lib/http/api";
import { toast } from "@/lib/toast";

import {
  buildBody,
  DEFAULT_POOL_NAME,
  EMPTY_WORKLOAD_CLASS,
  fairnessSettingsSchema,
  settingsToForm,
  type FairnessSettings,
  type FairnessSettingsFormValues,
  type FairnessSettingsResponse,
  type FairnessSettingsSubmitValues,
} from "./schema";

export const FAIRNESS_SETTINGS_QUERY_KEY = ["fairnessSettings"] as const;
export const FAIRNESS_STATUS_QUERY_KEY = ["fairnessStatus"] as const;

const defaultFetchSettings = async (): Promise<FairnessSettingsResponse> => {
  const { data } = await fetchClient.GET("/fairness/settings");
  if (data === undefined) throw new Error("Failed to load fairness settings");
  return data;
};

const defaultUpdateSettings = async (body: FairnessSettings): Promise<void> => {
  await fetchClient.PUT("/fairness/settings", { body });
};

type SettingsControl = Control<FairnessSettingsFormValues, unknown, FairnessSettingsSubmitValues>;

const NumberInput = ({
  control,
  name,
  label,
  description,
  step = "any",
  disabled,
}: {
  control: SettingsControl;
  name:
    | "default_reserved_share"
    | "default_max_queue_wait_seconds"
    | "saturation_threshold"
    | "saturation_check_cache_ttl"
    | "max_queue_depth_per_class"
    | "queue_poll_interval_seconds"
    | `workload_classes.${number}.reserved_share`
    | `workload_classes.${number}.max_queue_wait_seconds`;
  label: string;
  description?: string;
  step?: string;
  disabled: boolean;
}) => (
  <FormField control={control} name={name} label={label} description={description}>
    {({ ref, ...field }) => <Input {...field} ref={ref} type="number" step={step} disabled={disabled} />}
  </FormField>
);

const WorkloadClassesField = ({ control, disabled }: { control: SettingsControl; disabled: boolean }) => {
  const { fields, append, remove } = useFieldArray({ control, name: "workload_classes" });

  return (
    <div className="flex w-full flex-col gap-3">
      <div>
        <p className="text-sm font-medium">Workload classes</p>
        <p className="text-sm text-muted-foreground">
          Assign a class to a team or key by setting <code>metadata.priority</code> to the class name. Below the
          saturation threshold every class may borrow idle capacity; above it each class is held to its reserved share.
          Keys and teams without a class fall into the <code>{DEFAULT_POOL_NAME}</code> pool.
        </p>
      </div>

      {fields.map((field, index) => (
        <div key={field.id} className="rounded-lg border border-border p-4" data-testid={`workload-class-${index}`}>
          <div className="mb-3 flex items-center justify-between">
            <p className="text-sm font-medium">Class {index + 1}</p>
            <Button type="button" variant="destructive" size="sm" onClick={() => remove(index)} disabled={disabled}>
              Remove
            </Button>
          </div>
          <div className="grid grid-cols-1 gap-3 md:grid-cols-4">
            <FormField control={control} name={`workload_classes.${index}.name`} label="Name">
              {({ ref, ...nameField }) => (
                <Input {...nameField} ref={ref} placeholder="production" disabled={disabled} />
              )}
            </FormField>
            <NumberInput
              control={control}
              name={`workload_classes.${index}.reserved_share`}
              label="Reserved share (%)"
              disabled={disabled}
            />
            <NumberInput
              control={control}
              name={`workload_classes.${index}.max_queue_wait_seconds`}
              label="Max queue wait (s)"
              disabled={disabled}
            />
            <FormField control={control} name={`workload_classes.${index}.description`} label="Description">
              {({ ref, ...descriptionField }) => (
                <Input {...descriptionField} ref={ref} placeholder="Optional" disabled={disabled} />
              )}
            </FormField>
          </div>
        </div>
      ))}

      <Button type="button" variant="outline" onClick={() => append(EMPTY_WORKLOAD_CLASS)} disabled={disabled}>
        Add workload class
      </Button>
    </div>
  );
};

const ShareSummary = ({ control }: { control: SettingsControl }) => {
  const classes = useWatch({ control, name: "workload_classes" });
  const defaultShare = useWatch({ control, name: "default_reserved_share" });
  const total = classes.reduce((sum, workloadClass) => sum + (Number(workloadClass.reserved_share) || 0), 0);
  const withDefault = total + (Number(defaultShare) || 0);
  return (
    <p className="text-sm text-muted-foreground" data-testid="share-summary">
      Reserved in total: {Math.round(withDefault * 100) / 100}% of each model&apos;s RPM and TPM once saturated
    </p>
  );
};

interface SettingsFormProps {
  initialValues: FairnessSettingsFormValues;
  readOnly: boolean;
  updateSettings: (body: FairnessSettings) => Promise<void>;
}

const SettingsForm = ({ initialValues, readOnly, updateSettings }: SettingsFormProps) => {
  const queryClient = useQueryClient();
  const form = useZodForm(fairnessSettingsSchema, { defaultValues: initialValues });
  const { isDirty } = form.formState;

  const mutation = useMutation({
    mutationFn: (values: FairnessSettingsSubmitValues) => updateSettings(buildBody(values)),
    onSuccess: (_result, values) => {
      toast.success("Fairness settings saved");
      queryClient.invalidateQueries({ queryKey: FAIRNESS_SETTINGS_QUERY_KEY });
      queryClient.invalidateQueries({ queryKey: FAIRNESS_STATUS_QUERY_KEY });
      queryClient.invalidateQueries({ queryKey: WORKLOAD_CLASS_QUERY_KEY });
      form.reset(values);
    },
    onError: (error: unknown) => toast.fromError(error),
  });

  const onSubmit = form.handleSubmit((values) => mutation.mutate(values));
  const disabled = readOnly || mutation.isPending;

  return (
    <form onSubmit={onSubmit} noValidate>
      <FieldGroup>
        <FormField
          control={form.control}
          name="enabled"
          label="Enable fairness under load"
          description="Reserves estimated tokens before dispatch, protects each class's reserved share once a model saturates, and queues over-limit requests instead of rejecting them"
          orientation="horizontal"
        >
          {({ id, value, onChange, "aria-invalid": ariaInvalid, "aria-describedby": ariaDescribedBy }) => (
            <Switch
              id={id}
              checked={value}
              onCheckedChange={onChange}
              disabled={disabled}
              aria-invalid={ariaInvalid}
              aria-describedby={ariaDescribedBy}
            />
          )}
        </FormField>

        <WorkloadClassesField control={form.control} disabled={disabled} />

        <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
          <NumberInput
            control={form.control}
            name="default_reserved_share"
            label="Default pool reserved share (%)"
            description="Share kept for keys and teams without a workload class"
            disabled={disabled}
          />
          <NumberInput
            control={form.control}
            name="default_max_queue_wait_seconds"
            label="Default max queue wait (s)"
            description="Deadline for classes without their own; 0 rejects immediately. Clients may lower it per request with the x-litellm-max-queue-wait header"
            disabled={disabled}
          />
          <NumberInput
            control={form.control}
            name="saturation_threshold"
            label="Saturation threshold (%)"
            description="Model utilisation above which reserved shares are enforced"
            disabled={disabled}
          />
          <NumberInput
            control={form.control}
            name="saturation_check_cache_ttl"
            label="Saturation check cache (s)"
            description="How long a model's saturation reading is reused"
            step="1"
            disabled={disabled}
          />
          <NumberInput
            control={form.control}
            name="max_queue_depth_per_class"
            label="Max queue depth per class"
            description="Requests waiting per model and class before new ones are rejected with queue_full"
            step="1"
            disabled={disabled}
          />
          <NumberInput
            control={form.control}
            name="queue_poll_interval_seconds"
            label="Queue poll interval (s)"
            description="How often a queued request retries admission"
            disabled={disabled}
          />
        </div>
        <ShareSummary control={form.control} />
      </FieldGroup>

      {!readOnly && (
        <div className="mt-6 flex items-center justify-end gap-2">
          <Button
            type="button"
            variant="outline"
            onClick={() => form.reset(initialValues)}
            disabled={!isDirty || disabled}
          >
            Reset
          </Button>
          <Button type="submit" disabled={!isDirty || disabled}>
            {mutation.isPending ? "Saving..." : "Save settings"}
          </Button>
        </div>
      )}
    </form>
  );
};

const SettingsCard = ({ children }: { children: React.ReactNode }) => (
  <Card>
    <CardHeader>
      <CardTitle>Fairness under load</CardTitle>
      <CardDescription>
        Share each model&apos;s RPM and TPM across production, interactive, batch and development traffic without
        letting any class starve. Settings are stored in the database and take effect immediately on every worker.
      </CardDescription>
    </CardHeader>
    <CardContent>{children}</CardContent>
  </Card>
);

export interface FairnessSettingsFormProps {
  readOnly?: boolean;
  fetchSettings?: () => Promise<FairnessSettingsResponse>;
  updateSettings?: (body: FairnessSettings) => Promise<void>;
}

export const FairnessSettingsForm = ({
  readOnly = false,
  fetchSettings = defaultFetchSettings,
  updateSettings = defaultUpdateSettings,
}: FairnessSettingsFormProps) => {
  const { data, isPending, isError } = useQuery({ queryKey: FAIRNESS_SETTINGS_QUERY_KEY, queryFn: fetchSettings });
  const initialValues = React.useMemo(() => (data === undefined ? undefined : settingsToForm(data.settings)), [data]);

  if (isPending) {
    return (
      <SettingsCard>
        <Skeleton className="h-64 w-full" />
      </SettingsCard>
    );
  }

  if (isError || initialValues === undefined) {
    return (
      <SettingsCard>
        <p role="alert">Could not load the fairness settings.</p>
      </SettingsCard>
    );
  }

  return (
    <SettingsCard>
      {!data.persisted && (
        <p className="mb-4 text-sm text-muted-foreground">
          Showing the values from config.yaml. Saving here stores them in the database, which then takes precedence.
        </p>
      )}
      <SettingsForm
        key={JSON.stringify(initialValues)}
        initialValues={initialValues}
        readOnly={readOnly}
        updateSettings={updateSettings}
      />
    </SettingsCard>
  );
};
