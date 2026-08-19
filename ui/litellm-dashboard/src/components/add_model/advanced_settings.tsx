import React from "react";
import { Switch, Select, Tooltip, DatePicker } from "antd";
import { ChevronDown } from "lucide-react";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { Input } from "@/components/ui/input";
import { Row, Col, Typography } from "antd";
import TextArea from "antd/es/input/TextArea";
import { InfoCircleOutlined } from "@ant-design/icons";
import { Team } from "../key_team_helpers/key_list";
import { antdRules } from "../common_components/antdFormRules";
import { labelWithHint } from "../common_components/LabelWithHint";
import { MountedFormField } from "../common_components/MountedFormField";
import CacheControlInjectionPoints, {
  CACHE_CONTROL_LABEL,
  CACHE_CONTROL_TOOLTIP,
  NEW_CACHE_CONTROL_POINT,
} from "./cache_control_settings";
import VectorStoreSelector from "../vector_store_management/VectorStoreSelector";
import { Tag } from "../tag_management/types";
import { formItemValidateJSON } from "../../utils/textUtils";
import {
  PTU_COUNT_FIELD,
  PTU_RATE_FIELD,
  PTU_START_FIELD,
  ptuCountRules,
  ptuNoUsageCostRule,
  ptuPairRule,
  ptuRateRules,
  ptuStartRequiredRule,
  ptuWindowOrderRule,
  PTU_END_FIELD,
} from "../../utils/ptuValidation";
import { usePtuCostAttributionEnabled } from "@/app/(dashboard)/hooks/uiSettings/usePtuCostAttributionEnabled";
const { Link } = Typography;

interface AdvancedSettingsProps {
  showAdvancedSettings: boolean;
  setShowAdvancedSettings: (show: boolean) => void;
  teams?: Team[] | null;
  guardrailsList: string[];
  tagsList: Record<string, Tag>;
  accessToken: string;
}

const USAGE_COST_FIELDS = [
  "input_cost_per_token",
  "output_cost_per_token",
  "cache_read_input_token_cost",
  "cache_creation_input_token_cost",
  "input_cost_per_second",
];

// antd revalidates a field when one of its `dependencies` changes; react-hook-form drives the same
// edge from the other side, so each entry lists the fields to revalidate when THIS field changes.
const PTU_COUNT_DEPS = [PTU_RATE_FIELD, PTU_START_FIELD, ...USAGE_COST_FIELDS];

const validateNumber = (_: unknown, value: unknown) => {
  if (!value) {
    return Promise.resolve();
  }
  if (isNaN(Number(value)) || Number(value) < 0) {
    return Promise.reject("Please enter a valid positive number");
  }
  return Promise.resolve();
};

const usageCostRules = {
  deps: [PTU_COUNT_FIELD],
  validate: antdRules({ validator: validateNumber }, ptuNoUsageCostRule(PTU_COUNT_FIELD)),
};

const AdvancedSettings: React.FC<AdvancedSettingsProps> = ({
  showAdvancedSettings,
  setShowAdvancedSettings,
  teams,
  guardrailsList,
  tagsList,
  accessToken,
}) => {
  const [customPricing, setCustomPricing] = React.useState(false);
  const [pricingModel, setPricingModel] = React.useState<"per_token" | "per_second">("per_token");
  const [showCacheControl, setShowCacheControl] = React.useState(false);
  const ptuCostAttributionEnabled = usePtuCostAttributionEnabled();

  return (
    <>
      <Collapsible className="mt-2 mb-4 overflow-hidden rounded-lg border">
        <CollapsibleTrigger className="group/section flex w-full items-center justify-between px-4 py-3 text-left">
          <b>Advanced Settings</b>
          <ChevronDown className="size-5 shrink-0 text-muted-foreground transition-transform group-data-[panel-open]/section:rotate-180" />
        </CollapsibleTrigger>
        <CollapsibleContent className="px-4 pb-3">
          <div className="rounded-lg">
            <MountedFormField name="custom_pricing" label="Custom Pricing" className="mb-4">
              {(control) => (
                <Switch
                  id={control.id}
                  checked={control.value === true}
                  onChange={(checked) => {
                    control.onChange(checked);
                    setCustomPricing(checked);
                  }}
                  className="bg-gray-600"
                />
              )}
            </MountedFormField>

            <MountedFormField
              name="vector_store_ids"
              label={
                <span>
                  Attached Knowledge Bases (RAG){" "}
                  <Tooltip title="Vector stores to use for RAG. Every request to this model will automatically retrieve context from these knowledge bases.">
                    <a
                      href="https://docs.litellm.ai/docs/completion/knowledgebase"
                      target="_blank"
                      rel="noopener noreferrer"
                      onClick={(e) => e.stopPropagation()}
                    >
                      <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                    </a>
                  </Tooltip>
                </span>
              }
              className="mt-4"
              help="Select vector stores to attach. Requests to this model will automatically use these for RAG. Set up vector stores in Tools > Vector Stores."
            >
              {(control) => (
                <VectorStoreSelector
                  onChange={control.onChange}
                  value={control.value as string[] | undefined}
                  accessToken={accessToken}
                  placeholder="Select knowledge bases (optional)"
                />
              )}
            </MountedFormField>

            <MountedFormField
              name="guardrails"
              label={
                <span>
                  Guardrails{" "}
                  <Tooltip title="Apply safety guardrails to this key to filter content or enforce policies">
                    <a
                      href="https://docs.litellm.ai/docs/proxy/guardrails/quick_start"
                      target="_blank"
                      rel="noopener noreferrer"
                      onClick={(e) => e.stopPropagation()} // Prevent accordion from collapsing when clicking link
                    >
                      <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                    </a>
                  </Tooltip>
                </span>
              }
              className="mt-4"
              help="Select existing guardrails. Go to 'Guardrails' tab to create new guardrails."
            >
              {(control) => (
                <Select
                  id={control.id}
                  mode="tags"
                  style={{ width: "100%" }}
                  placeholder="Select or enter guardrails"
                  value={control.value as string[] | undefined}
                  onChange={control.onChange}
                  onBlur={control.onBlur}
                  options={guardrailsList.map((name) => ({ value: name, label: name }))}
                />
              )}
            </MountedFormField>

            <MountedFormField name="tags" label="Tags" className="mb-4">
              {(control) => (
                <Select
                  id={control.id}
                  mode="tags"
                  style={{ width: "100%" }}
                  placeholder="Select or enter tags"
                  value={control.value as string[] | undefined}
                  onChange={control.onChange}
                  onBlur={control.onBlur}
                  options={Object.values(tagsList).map((tag) => ({
                    value: tag.name,
                    label: tag.name,
                    title: tag.description || tag.name,
                  }))}
                />
              )}
            </MountedFormField>

            {ptuCostAttributionEnabled && (
              <>
                <MountedFormField
                  name={PTU_COUNT_FIELD}
                  label={labelWithHint(
                    "PTU Count",
                    "Provisioned throughput units for this deployment. Set together with Cost per PTU / Hour and a Team to attribute a flat daily cost.",
                  )}
                  rules={{
                    deps: PTU_COUNT_DEPS,
                    validate: antdRules({ validator: validateNumber }, ...ptuCountRules, ptuPairRule(PTU_RATE_FIELD)),
                  }}
                  className="mb-4"
                >
                  {(control) => (
                    <Input
                      id={control.id}
                      value={(control.value as string | undefined) ?? ""}
                      onChange={control.onChange}
                      onBlur={control.onBlur}
                      placeholder="e.g. 15"
                    />
                  )}
                </MountedFormField>

                <MountedFormField
                  name={PTU_RATE_FIELD}
                  label={labelWithHint(
                    "Calculated Cost per PTU / Hour (USD)",
                    "Flat cost = PTU count * this rate * active hours, attributed to the deployment's team.",
                  )}
                  rules={{
                    deps: [PTU_COUNT_FIELD],
                    validate: antdRules({ validator: validateNumber }, ...ptuRateRules, ptuPairRule(PTU_COUNT_FIELD)),
                  }}
                  className="mb-4"
                >
                  {(control) => (
                    <Input
                      id={control.id}
                      value={(control.value as string | undefined) ?? ""}
                      onChange={control.onChange}
                      onBlur={control.onBlur}
                      placeholder="e.g. 2.00"
                    />
                  )}
                </MountedFormField>

                <MountedFormField
                  name={PTU_START_FIELD}
                  label={labelWithHint(
                    "PTU Effective From (UTC)",
                    "Start of the PTU window, required when PTU Count is set. Flat cost accrues by the hour within the window; a window opening at 23:00 charges one hour that day.",
                  )}
                  rules={{
                    deps: [PTU_END_FIELD],
                    validate: antdRules(
                      ptuStartRequiredRule(PTU_COUNT_FIELD),
                      ptuWindowOrderRule(PTU_END_FIELD, "start"),
                    ),
                  }}
                  className="mb-4"
                >
                  {(control) => (
                    <DatePicker
                      id={control.id}
                      showTime
                      style={{ width: "100%" }}
                      value={control.value as never}
                      onChange={control.onChange}
                      onBlur={control.onBlur}
                    />
                  )}
                </MountedFormField>

                <MountedFormField
                  name={PTU_END_FIELD}
                  label={labelWithHint(
                    "PTU Effective To (UTC)",
                    "Optional end of the PTU window (exclusive). Leave blank for open-ended.",
                  )}
                  rules={{
                    deps: [PTU_START_FIELD],
                    validate: antdRules(ptuWindowOrderRule(PTU_START_FIELD, "end")),
                  }}
                  className="mb-4"
                >
                  {(control) => (
                    <DatePicker
                      id={control.id}
                      showTime
                      style={{ width: "100%" }}
                      value={control.value as never}
                      onChange={control.onChange}
                      onBlur={control.onBlur}
                    />
                  )}
                </MountedFormField>
              </>
            )}

            {customPricing && (
              <div className="ml-6 pl-4 border-l-2 border-border">
                <MountedFormField name="pricing_model" label="Pricing Model" className="mb-4">
                  {(control) => (
                    <Select
                      id={control.id}
                      defaultValue="per_token"
                      value={control.value as "per_token" | "per_second" | undefined}
                      onBlur={control.onBlur}
                      onChange={(value: "per_token" | "per_second") => {
                        control.onChange(value);
                        setPricingModel(value);
                      }}
                      options={[
                        { value: "per_token", label: "Per Million Tokens" },
                        { value: "per_second", label: "Per Second" },
                      ]}
                    />
                  )}
                </MountedFormField>

                {pricingModel === "per_token" ? (
                  <>
                    <MountedFormField
                      name="input_cost_per_token"
                      label="Input Cost (per 1M tokens)"
                      rules={usageCostRules}
                      className="mb-4"
                    >
                      {(control) => (
                        <Input
                          id={control.id}
                          value={(control.value as string | undefined) ?? ""}
                          onChange={control.onChange}
                          onBlur={control.onBlur}
                        />
                      )}
                    </MountedFormField>
                    <MountedFormField
                      name="output_cost_per_token"
                      label="Output Cost (per 1M tokens)"
                      rules={usageCostRules}
                      className="mb-4"
                    >
                      {(control) => (
                        <Input
                          id={control.id}
                          value={(control.value as string | undefined) ?? ""}
                          onChange={control.onChange}
                          onBlur={control.onBlur}
                        />
                      )}
                    </MountedFormField>
                    <MountedFormField
                      name="cache_read_input_token_cost"
                      label={labelWithHint("Cache Read Cost (per 1M tokens)", "If left blank, defaults to Input Cost.")}
                      rules={usageCostRules}
                      className="mb-4"
                    >
                      {(control) => (
                        <Input
                          id={control.id}
                          value={(control.value as string | undefined) ?? ""}
                          onChange={control.onChange}
                          onBlur={control.onBlur}
                          placeholder="Defaults to Input Cost if blank"
                        />
                      )}
                    </MountedFormField>
                    <MountedFormField
                      name="cache_creation_input_token_cost"
                      label={labelWithHint(
                        "Cache Write Cost (per 1M tokens)",
                        "If left blank, defaults to Input Cost (the backend falls back to input_cost_per_token when no cache-write rate is set).",
                      )}
                      rules={usageCostRules}
                      className="mb-4"
                    >
                      {(control) => (
                        <Input
                          id={control.id}
                          value={(control.value as string | undefined) ?? ""}
                          onChange={control.onChange}
                          onBlur={control.onBlur}
                          placeholder="Defaults to Input Cost if blank"
                        />
                      )}
                    </MountedFormField>
                  </>
                ) : (
                  <MountedFormField
                    name="input_cost_per_second"
                    label="Cost Per Second"
                    rules={usageCostRules}
                    className="mb-4"
                  >
                    {(control) => (
                      <Input
                        id={control.id}
                        value={(control.value as string | undefined) ?? ""}
                        onChange={control.onChange}
                        onBlur={control.onBlur}
                      />
                    )}
                  </MountedFormField>
                )}
              </div>
            )}

            <MountedFormField
              name="use_in_pass_through"
              label={labelWithHint(
                "Use in pass through routes",
                <span>
                  Allow using these credentials in pass through routes.{" "}
                  <Link href="https://docs.litellm.ai/docs/pass_through/vertex_ai" target="_blank">
                    Learn more
                  </Link>
                </span>,
              )}
              className="mb-4 mt-4"
            >
              {(control) => (
                <Switch
                  id={control.id}
                  checked={control.value === true}
                  onChange={control.onChange}
                  className="bg-gray-600"
                />
              )}
            </MountedFormField>

            <MountedFormField
              name="cache_control"
              label={labelWithHint(CACHE_CONTROL_LABEL, CACHE_CONTROL_TOOLTIP)}
              className="mb-4"
            >
              {(control) => (
                <Switch
                  id={control.id}
                  checked={control.value === true}
                  onChange={(checked) => {
                    control.onChange(checked);
                    setShowCacheControl(checked);
                  }}
                  className="bg-gray-600"
                />
              )}
            </MountedFormField>

            {showCacheControl && (
              <MountedFormField name="cache_control_injection_points" defaultValue={[NEW_CACHE_CONTROL_POINT]} bare>
                {(control) => (
                  <CacheControlInjectionPoints
                    value={control.value as React.ComponentProps<typeof CacheControlInjectionPoints>["value"]}
                    onChange={control.onChange}
                  />
                )}
              </MountedFormField>
            )}
            <MountedFormField
              name="litellm_extra_params"
              label={labelWithHint(
                "LiteLLM Params",
                "Optional litellm params used for making a litellm.completion() call.",
              )}
              className="mb-4 mt-4"
              rules={{ validate: antdRules({ validator: formItemValidateJSON }) }}
            >
              {(control) => (
                <TextArea
                  id={control.id}
                  value={(control.value as string | undefined) ?? ""}
                  onChange={control.onChange}
                  onBlur={control.onBlur}
                  rows={4}
                  placeholder='{
                  "rpm": 100,
                  "timeout": 0,
                  "stream_timeout": 0
                }'
                />
              )}
            </MountedFormField>
            <Row className="mb-4">
              <Col span={10}></Col>
              <Col span={10}>
                <p className="text-muted-foreground text-sm">
                  Pass JSON of litellm supported params{" "}
                  <Link href="https://docs.litellm.ai/docs/completion/input" target="_blank">
                    litellm.completion() call
                  </Link>
                </p>
              </Col>
            </Row>
            <MountedFormField
              name="model_info_params"
              label={labelWithHint(
                "Model Info",
                "Optional model info params. Returned when calling `/model/info` endpoint.",
              )}
              className="mb-0"
              rules={{ validate: antdRules({ validator: formItemValidateJSON }) }}
            >
              {(control) => (
                <TextArea
                  id={control.id}
                  value={(control.value as string | undefined) ?? ""}
                  onChange={control.onChange}
                  onBlur={control.onBlur}
                  rows={4}
                  placeholder='{
                  "mode": "chat"
                }'
                />
              )}
            </MountedFormField>
          </div>
        </CollapsibleContent>
      </Collapsible>
    </>
  );
};

export default AdvancedSettings;
