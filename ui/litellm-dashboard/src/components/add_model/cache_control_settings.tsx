import React from "react";
import {
  useFieldArray,
  useFormContext,
  Controller,
  FormProvider,
  useForm,
} from "react-hook-form";
import { MinusCircle, Plus } from "lucide-react";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Input } from "@/components/ui/input";

interface CacheControlInjectionPoint {
  location: "message";
  role?: "user" | "system" | "assistant" | "";
  index?: number | null;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type AntdLikeFormInstance = any;

interface CacheControlSettingsProps {
  /**
   * Optional antd-style `FormInstance`. When provided (legacy consumers
   * such as `model_info_view.tsx`), the component self-wraps in its own
   * react-hook-form `FormProvider` so the controls keep working; the
   * submit wiring stays on the antd side via `getFieldValue` /
   * `setFieldValue`. When omitted (phase-1 shadcn callers), we use the
   * ambient RHF context via `useFormContext`.
   */
  form?: AntdLikeFormInstance;
  showCacheControl: boolean;
  onCacheControlChange: (checked: boolean) => void;
}

const CacheControlSettings: React.FC<CacheControlSettingsProps> = ({
  form,
  showCacheControl,
  onCacheControlChange,
}) => {
  if (form && typeof form.getFieldValue === "function") {
    return (
      <LegacyAntdCacheControlSettings
        form={form}
        showCacheControl={showCacheControl}
        onCacheControlChange={onCacheControlChange}
      />
    );
  }
  return (
    <RHFCacheControlSettings
      showCacheControl={showCacheControl}
      onCacheControlChange={onCacheControlChange}
    />
  );
};

interface RHFCacheControlSettingsProps {
  showCacheControl: boolean;
  onCacheControlChange: (checked: boolean) => void;
}

const RHFCacheControlSettings: React.FC<RHFCacheControlSettingsProps> = ({
  showCacheControl,
  onCacheControlChange,
}) => {
  const { control, getValues, setValue } = useFormContext();

  const { fields, append, remove } = useFieldArray({
    control,
    name: "cache_control_injection_points",
  });

  const syncExtraParams = () => {
    const injectionPoints = (getValues("cache_control_injection_points") ||
      []) as CacheControlInjectionPoint[];
    const cleaned = injectionPoints
      .filter((p) => p && (p.role || p.index !== undefined))
      .map((p) => {
        const next: Record<string, unknown> = {
          location: p.location ?? "message",
        };
        if (p.role) next.role = p.role;
        if (p.index !== undefined && p.index !== null) next.index = p.index;
        return next;
      });

    const currentParams = getValues("litellm_extra_params");
    try {
      const paramsObj = currentParams ? JSON.parse(currentParams) : {};
      if (cleaned.length > 0) {
        paramsObj.cache_control_injection_points = cleaned;
      } else {
        delete paramsObj.cache_control_injection_points;
      }
      if (Object.keys(paramsObj).length > 0) {
        setValue("litellm_extra_params", JSON.stringify(paramsObj, null, 2));
      } else {
        setValue("litellm_extra_params", "");
      }
    } catch (error) {
      console.error("Error updating cache control points:", error);
    }
  };

  return (
    <>
      <div className="grid grid-cols-24 gap-2 mb-4 items-center">
        <Label
          className="col-span-10"
          title="Tell litellm where to inject cache control checkpoints. You can specify either by role (to apply to all messages of that role) or by specific message index."
        >
          Cache Control Injection Points
        </Label>
        <div className="col-span-14">
          <Switch
            checked={showCacheControl}
            onCheckedChange={(checked) => {
              onCacheControlChange(!!checked);
            }}
          />
        </div>
      </div>

      {showCacheControl && (
        <div className="ml-6 pl-4 border-l-2 border-border">
          <p className="text-sm text-muted-foreground block mb-4">
            Providers like Anthropic, Bedrock API require users to specify
            where to inject cache control checkpoints, litellm can
            automatically add them for you as a cost saving feature.
          </p>

          {fields.map((field, index) => (
            <div
              key={field.id}
              className="flex items-center mb-4 gap-4 flex-wrap"
            >
              <div className="flex flex-col gap-1" style={{ width: "180px" }}>
                <Label>Type</Label>
                <Controller
                  control={control}
                  name={`cache_control_injection_points.${index}.location`}
                  defaultValue="message"
                  render={({ field: locationField }) => (
                    <Select
                      value={(locationField.value as string) || "message"}
                      onValueChange={locationField.onChange}
                      disabled
                    >
                      <SelectTrigger>
                        <SelectValue placeholder="Message" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="message">Message</SelectItem>
                      </SelectContent>
                    </Select>
                  )}
                />
              </div>

              <div className="flex flex-col gap-1" style={{ width: "180px" }}>
                <Label title="LiteLLM will mark all messages of this role as cacheable">
                  Role
                </Label>
                <Controller
                  control={control}
                  name={`cache_control_injection_points.${index}.role`}
                  render={({ field: roleField }) => (
                    <Select
                      value={(roleField.value as string) || ""}
                      onValueChange={(v) => {
                        roleField.onChange(v);
                        syncExtraParams();
                      }}
                    >
                      <SelectTrigger>
                        <SelectValue placeholder="Select a role" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="user">User</SelectItem>
                        <SelectItem value="system">System</SelectItem>
                        <SelectItem value="assistant">Assistant</SelectItem>
                      </SelectContent>
                    </Select>
                  )}
                />
              </div>

              <div className="flex flex-col gap-1" style={{ width: "180px" }}>
                <Label title="(Optional) If set litellm will mark the message at this index as cacheable">
                  Index
                </Label>
                <Controller
                  control={control}
                  name={`cache_control_injection_points.${index}.index`}
                  render={({ field: indexField }) => (
                    <Input
                      type="number"
                      step={1}
                      placeholder="Optional"
                      value={
                        indexField.value === null ||
                        indexField.value === undefined
                          ? ""
                          : (indexField.value as number)
                      }
                      onChange={(e) => {
                        const raw = e.target.value;
                        indexField.onChange(raw === "" ? null : Number(raw));
                        syncExtraParams();
                      }}
                      onWheel={(event) =>
                        (event.currentTarget as HTMLInputElement).blur()
                      }
                    />
                  )}
                />
              </div>

              {fields.length > 1 && (
                <button
                  type="button"
                  className="text-destructive cursor-pointer ml-12"
                  aria-label="Remove injection point"
                  onClick={() => {
                    remove(index);
                    setTimeout(() => {
                      syncExtraParams();
                    }, 0);
                  }}
                >
                  <MinusCircle className="h-5 w-5" />
                </button>
              )}
            </div>
          ))}

          <button
            type="button"
            className="flex items-center justify-center w-full border border-dashed border-border py-2 px-4 text-muted-foreground hover:text-primary hover:border-primary/50 transition-all rounded"
            onClick={() =>
              append({ location: "message", role: "", index: null })
            }
          >
            <Plus className="mr-2 h-4 w-4" />
            Add Injection Point
          </button>
        </div>
      )}
    </>
  );
};

/**
 * Legacy wrapper used by consumers that still drive the form with an antd
 * `FormInstance` (e.g. `model_info_view.tsx`). Bridges the antd form's
 * `getFieldValue` / `setFieldValue` helpers to the RHF-based UI.
 */
interface LegacyAntdCacheControlSettingsProps {
  form: AntdLikeFormInstance;
  showCacheControl: boolean;
  onCacheControlChange: (checked: boolean) => void;
}

const LegacyAntdCacheControlSettings: React.FC<
  LegacyAntdCacheControlSettingsProps
> = ({ form, showCacheControl, onCacheControlChange }) => {
  const rhf = useForm<{
    cache_control_injection_points: CacheControlInjectionPoint[];
  }>({
    defaultValues: {
      cache_control_injection_points:
        (form?.getFieldValue?.("cache_control_injection_points") as
          | CacheControlInjectionPoint[]
          | undefined) ?? [{ location: "message" }],
    },
  });

  // Mirror RHF state back into the antd form as it changes so submit wiring
  // keeps working without rewriting the parent.
  React.useEffect(() => {
    const subscription = rhf.watch((values) => {
      if (form?.setFieldValue) {
        form.setFieldValue(
          "cache_control_injection_points",
          values.cache_control_injection_points,
        );
      }
      // Also sync to litellm_extra_params the same way the non-legacy flow
      // does, if the caller wires through that field. This matches the
      // prior antd implementation's behavior.
      if (
        form?.getFieldValue &&
        form?.setFieldValue &&
        typeof form.getFieldValue === "function"
      ) {
        const currentParams = form.getFieldValue("litellm_extra_params");
        try {
          const paramsObj = currentParams ? JSON.parse(currentParams) : {};
          const points = (values.cache_control_injection_points ||
            []) as CacheControlInjectionPoint[];
          const cleaned = points
            .filter((p) => p && (p.role || p.index !== undefined))
            .map((p) => {
              const next: Record<string, unknown> = {
                location: p.location ?? "message",
              };
              if (p.role) next.role = p.role;
              if (p.index !== undefined && p.index !== null)
                next.index = p.index;
              return next;
            });
          if (cleaned.length > 0) {
            paramsObj.cache_control_injection_points = cleaned;
          } else {
            delete paramsObj.cache_control_injection_points;
          }
          if (Object.keys(paramsObj).length > 0) {
            form.setFieldValue(
              "litellm_extra_params",
              JSON.stringify(paramsObj, null, 2),
            );
          } else {
            form.setFieldValue("litellm_extra_params", "");
          }
        } catch {
          /* best-effort — caller may not track litellm_extra_params */
        }
      }
    });
    return () => subscription.unsubscribe();
  }, [rhf, form]);

  return (
    <FormProvider {...rhf}>
      <RHFCacheControlSettings
        showCacheControl={showCacheControl}
        onCacheControlChange={onCacheControlChange}
      />
    </FormProvider>
  );
};

export default CacheControlSettings;
