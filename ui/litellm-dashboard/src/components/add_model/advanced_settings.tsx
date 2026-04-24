import React from "react";
import { Controller, useFormContext, useWatch } from "react-hook-form";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Info as InfoCircleOutlined, X } from "lucide-react";
import { Team } from "../key_team_helpers/key_list";
import CacheControlSettings from "./cache_control_settings";
import VectorStoreSelector from "../vector_store_management/VectorStoreSelector";
import { Tag } from "../tag_management/types";
import { validateJsonValue } from "../../utils/textUtils";

interface AdvancedSettingsProps {
  showAdvancedSettings: boolean;
  setShowAdvancedSettings: (show: boolean) => void;
  teams?: Team[] | null;
  guardrailsList: string[];
  tagsList: Record<string, Tag>;
  accessToken: string;
}

function TagMultiSelect({
  value,
  onChange,
  options,
  placeholder,
}: {
  value: string[];
  onChange: (next: string[]) => void;
  options: { value: string; label: string }[];
  placeholder: string;
}) {
  const selected = value ?? [];
  const remaining = options.filter(
    (o) => !selected.includes(o.value as string),
  );
  const [input, setInput] = React.useState("");

  const addValue = (next: string) => {
    const trimmed = next.trim();
    if (!trimmed) return;
    if (selected.includes(trimmed)) return;
    onChange([...selected, trimmed]);
  };

  return (
    <div className="space-y-2">
      <div className="flex gap-2">
        <Input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder={placeholder}
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
            if (v) {
              addValue(v);
            }
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
                <SelectItem key={opt.value} value={opt.value}>
                  {opt.label}
                </SelectItem>
              ))
            )}
          </SelectContent>
        </Select>
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

const AdvancedSettings: React.FC<AdvancedSettingsProps> = ({
  guardrailsList,
  tagsList,
  accessToken,
}) => {
  const { control, getValues, setValue, formState } = useFormContext();
  const customPricing = !!useWatch({ control, name: "custom_pricing" });
  const pricingModel =
    (useWatch({ control, name: "pricing_model" }) as
      | "per_token"
      | "per_second"
      | undefined) ?? "per_token";
  const showCacheControl = !!useWatch({ control, name: "cache_control" });

  const handleCustomPricingChange = (checked: boolean) => {
    setValue("custom_pricing", checked);
    if (!checked) {
      setValue("input_cost_per_token", undefined);
      setValue("output_cost_per_token", undefined);
      setValue("input_cost_per_second", undefined);
    }
  };

  const handlePassThroughChange = (checked: boolean) => {
    setValue("use_in_pass_through", checked);
    const currentParams = getValues("litellm_extra_params") as
      | string
      | undefined;
    try {
      const paramsObj = currentParams ? JSON.parse(currentParams) : {};
      if (checked) {
        paramsObj.use_in_pass_through = true;
      } else {
        delete paramsObj.use_in_pass_through;
      }
      if (Object.keys(paramsObj).length > 0) {
        setValue(
          "litellm_extra_params",
          JSON.stringify(paramsObj, null, 2),
        );
      } else {
        setValue("litellm_extra_params", "");
      }
    } catch {
      if (checked) {
        setValue(
          "litellm_extra_params",
          JSON.stringify({ use_in_pass_through: true }, null, 2),
        );
      } else {
        setValue("litellm_extra_params", "");
      }
    }
  };

  const handleCacheControlChange = (checked: boolean) => {
    setValue("cache_control", checked);
    if (!checked) {
      const currentParams = getValues("litellm_extra_params") as
        | string
        | undefined;
      try {
        const paramsObj = currentParams ? JSON.parse(currentParams) : {};
        delete paramsObj.cache_control_injection_points;
        if (Object.keys(paramsObj).length > 0) {
          setValue(
            "litellm_extra_params",
            JSON.stringify(paramsObj, null, 2),
          );
        } else {
          setValue("litellm_extra_params", "");
        }
      } catch {
        setValue("litellm_extra_params", "");
      }
    }
  };

  const validateNumber = (value: unknown): true | string => {
    if (value === undefined || value === null || value === "") {
      return true;
    }
    const num = Number(value);
    if (Number.isNaN(num) || num < 0) {
      return "Please enter a valid positive number";
    }
    return true;
  };

  const errors = formState.errors as Record<
    string,
    { message?: string } | undefined
  >;

  return (
    <Accordion type="single" collapsible className="mt-2 mb-4">
      <AccordionItem value="advanced-settings">
        <AccordionTrigger>
          <b>Advanced Settings</b>
        </AccordionTrigger>
        <AccordionContent>
          <div className="bg-background rounded-lg">
            <div className="grid grid-cols-24 gap-2 mb-4 items-center">
              <Label className="col-span-10">Custom Pricing</Label>
              <div className="col-span-14">
                <Controller
                  control={control}
                  name="custom_pricing"
                  render={({ field }) => (
                    <Switch
                      checked={!!field.value}
                      onCheckedChange={(checked) => {
                        field.onChange(checked);
                        handleCustomPricingChange(checked);
                      }}
                    />
                  )}
                />
              </div>
            </div>

            <div className="grid grid-cols-24 gap-2 mb-4 items-start">
              <Label className="col-span-10 pt-2">
                Attached Knowledge Bases (RAG){" "}
                <span
                  className="ml-1"
                  title="Vector stores to use for RAG. Every request to this model will automatically retrieve context from these knowledge bases."
                >
                  <a
                    href="https://docs.litellm.ai/docs/completion/knowledgebase"
                    target="_blank"
                    rel="noopener noreferrer"
                    onClick={(e) => e.stopPropagation()}
                  >
                    <InfoCircleOutlined
                      className="h-3.5 w-3.5 inline-block"
                      style={{ marginLeft: "4px" }}
                    />
                  </a>
                </span>
              </Label>
              <div className="col-span-14">
                <Controller
                  control={control}
                  name="vector_store_ids"
                  defaultValue={[]}
                  render={({ field }) => (
                    <VectorStoreSelector
                      value={(field.value as string[]) ?? []}
                      onChange={field.onChange}
                      accessToken={accessToken}
                      placeholder="Select knowledge bases (optional)"
                    />
                  )}
                />
                <p className="text-sm text-muted-foreground mt-1">
                  Select vector stores to attach. Requests to this model will
                  automatically use these for RAG. Set up vector stores in
                  Tools &gt; Vector Stores.
                </p>
              </div>
            </div>

            <div className="grid grid-cols-24 gap-2 mb-4 items-start">
              <Label className="col-span-10 pt-2">
                Guardrails{" "}
                <span
                  className="ml-1"
                  title="Apply safety guardrails to this key to filter content or enforce policies"
                >
                  <a
                    href="https://docs.litellm.ai/docs/proxy/guardrails/quick_start"
                    target="_blank"
                    rel="noopener noreferrer"
                    onClick={(e) => e.stopPropagation()}
                  >
                    <InfoCircleOutlined
                      className="h-3.5 w-3.5 inline-block"
                      style={{ marginLeft: "4px" }}
                    />
                  </a>
                </span>
              </Label>
              <div className="col-span-14">
                <Controller
                  control={control}
                  name="guardrails"
                  defaultValue={[]}
                  render={({ field }) => (
                    <TagMultiSelect
                      value={(field.value as string[]) ?? []}
                      onChange={field.onChange}
                      placeholder="Select or enter guardrails"
                      options={guardrailsList.map((name) => ({
                        value: name,
                        label: name,
                      }))}
                    />
                  )}
                />
                <p className="text-sm text-muted-foreground mt-1">
                  Select existing guardrails. Go to &apos;Guardrails&apos; tab
                  to create new guardrails.
                </p>
              </div>
            </div>

            <div className="grid grid-cols-24 gap-2 mb-4 items-start">
              <Label className="col-span-10 pt-2">Tags</Label>
              <div className="col-span-14">
                <Controller
                  control={control}
                  name="tags"
                  defaultValue={[]}
                  render={({ field }) => (
                    <TagMultiSelect
                      value={(field.value as string[]) ?? []}
                      onChange={field.onChange}
                      placeholder="Select or enter tags"
                      options={Object.values(tagsList).map((tag) => ({
                        value: tag.name,
                        label: tag.name,
                      }))}
                    />
                  )}
                />
              </div>
            </div>

            {customPricing && (
              <div className="ml-6 pl-4 border-l-2 border-border">
                <div className="grid grid-cols-24 gap-2 mb-4 items-center">
                  <Label className="col-span-10">Pricing Model</Label>
                  <div className="col-span-14">
                    <Controller
                      control={control}
                      name="pricing_model"
                      defaultValue="per_token"
                      render={({ field }) => (
                        <Select
                          value={(field.value as string) || "per_token"}
                          onValueChange={field.onChange}
                        >
                          <SelectTrigger>
                            <SelectValue placeholder="Pricing model" />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="per_token">
                              Per Million Tokens
                            </SelectItem>
                            <SelectItem value="per_second">
                              Per Second
                            </SelectItem>
                          </SelectContent>
                        </Select>
                      )}
                    />
                  </div>
                </div>

                {pricingModel === "per_token" ? (
                  <>
                    <div className="grid grid-cols-24 gap-2 mb-4 items-start">
                      <Label className="col-span-10 pt-2">
                        Input Cost (per 1M tokens)
                      </Label>
                      <div className="col-span-14">
                        <Controller
                          control={control}
                          name="input_cost_per_token"
                          rules={{ validate: validateNumber }}
                          render={({ field }) => (
                            <Input
                              type="text"
                              value={(field.value as string) ?? ""}
                              onChange={(e) => field.onChange(e.target.value)}
                            />
                          )}
                        />
                        {errors.input_cost_per_token?.message && (
                          <p className="text-sm text-destructive mt-1">
                            {String(errors.input_cost_per_token.message)}
                          </p>
                        )}
                      </div>
                    </div>
                    <div className="grid grid-cols-24 gap-2 mb-4 items-start">
                      <Label className="col-span-10 pt-2">
                        Output Cost (per 1M tokens)
                      </Label>
                      <div className="col-span-14">
                        <Controller
                          control={control}
                          name="output_cost_per_token"
                          rules={{ validate: validateNumber }}
                          render={({ field }) => (
                            <Input
                              type="text"
                              value={(field.value as string) ?? ""}
                              onChange={(e) => field.onChange(e.target.value)}
                            />
                          )}
                        />
                        {errors.output_cost_per_token?.message && (
                          <p className="text-sm text-destructive mt-1">
                            {String(errors.output_cost_per_token.message)}
                          </p>
                        )}
                      </div>
                    </div>
                  </>
                ) : (
                  <div className="grid grid-cols-24 gap-2 mb-4 items-start">
                    <Label className="col-span-10 pt-2">Cost Per Second</Label>
                    <div className="col-span-14">
                      <Controller
                        control={control}
                        name="input_cost_per_second"
                        rules={{ validate: validateNumber }}
                        render={({ field }) => (
                          <Input
                            type="text"
                            value={(field.value as string) ?? ""}
                            onChange={(e) => field.onChange(e.target.value)}
                          />
                        )}
                      />
                      {errors.input_cost_per_second?.message && (
                        <p className="text-sm text-destructive mt-1">
                          {String(errors.input_cost_per_second.message)}
                        </p>
                      )}
                    </div>
                  </div>
                )}
              </div>
            )}

            <div className="grid grid-cols-24 gap-2 mb-4 mt-4 items-center">
              <Label
                className="col-span-10"
                title="Allow using these credentials in pass through routes."
              >
                Use in pass through routes
              </Label>
              <div className="col-span-14">
                <Controller
                  control={control}
                  name="use_in_pass_through"
                  render={({ field }) => (
                    <Switch
                      checked={!!field.value}
                      onCheckedChange={(checked) => {
                        field.onChange(checked);
                        handlePassThroughChange(checked);
                      }}
                    />
                  )}
                />
              </div>
            </div>

            <CacheControlSettings
              showCacheControl={showCacheControl}
              onCacheControlChange={handleCacheControlChange}
            />

            <div className="grid grid-cols-24 gap-2 mb-4 mt-4 items-start">
              <Label
                className="col-span-10 pt-2"
                title="Optional litellm params used for making a litellm.completion() call."
              >
                LiteLLM Params
              </Label>
              <div className="col-span-14">
                <Controller
                  control={control}
                  name="litellm_extra_params"
                  rules={{ validate: validateJsonValue }}
                  render={({ field }) => (
                    <Textarea
                      rows={4}
                      placeholder='{
                  "rpm": 100,
                  "timeout": 0,
                  "stream_timeout": 0
                }'
                      value={(field.value as string) ?? ""}
                      onChange={field.onChange}
                    />
                  )}
                />
                {errors.litellm_extra_params?.message && (
                  <p className="text-sm text-destructive mt-1">
                    {String(errors.litellm_extra_params.message)}
                  </p>
                )}
              </div>
            </div>

            <div className="grid grid-cols-24 gap-2 mb-4 items-start">
              <div className="col-span-10" />
              <div className="col-span-14">
                <p className="text-muted-foreground text-sm">
                  Pass JSON of litellm supported params{" "}
                  <a
                    href="https://docs.litellm.ai/docs/completion/input"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-primary hover:text-primary/80 underline"
                  >
                    litellm.completion() call
                  </a>
                </p>
              </div>
            </div>

            <div className="grid grid-cols-24 gap-2 mb-0 items-start">
              <Label
                className="col-span-10 pt-2"
                title="Optional model info params. Returned when calling `/model/info` endpoint."
              >
                Model Info
              </Label>
              <div className="col-span-14">
                <Controller
                  control={control}
                  name="model_info_params"
                  rules={{ validate: validateJsonValue }}
                  render={({ field }) => (
                    <Textarea
                      rows={4}
                      placeholder='{
                  "mode": "chat"
                }'
                      value={(field.value as string) ?? ""}
                      onChange={field.onChange}
                    />
                  )}
                />
                {errors.model_info_params?.message && (
                  <p className="text-sm text-destructive mt-1">
                    {String(errors.model_info_params.message)}
                  </p>
                )}
              </div>
            </div>
          </div>
        </AccordionContent>
      </AccordionItem>
    </Accordion>
  );
};

export default AdvancedSettings;
