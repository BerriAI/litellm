import type { TFunction } from "i18next";
import { z } from "zod";
import { asText } from "./GuardrailFormField";
import {
  guardrail_provider_map,
  shouldRenderContentFilterConfigSettings,
  shouldRenderLLMJudgeFields,
} from "./guardrail_info_helpers";
import {
  isBlankGuardrailValue,
  validateOptionalParams,
  validateProviderFields,
  type GuardrailProviderParamsMap,
} from "./guardrailFormValidation";

interface AddGuardrailSchemaOptions {
  selectedProvider: string | null;
  providerParams: GuardrailProviderParamsMap | null;
  currentStep: number;
  isDecisionModelProvider: boolean;
  isToolPermissionProvider: boolean;
  t: TFunction;
}

const validateLlmJudgeFields = (values: Record<string, unknown>, t: TFunction, ctx: z.RefinementCtx) => {
  if (isBlankGuardrailValue(values.judge_model)) {
    ctx.addIssue({
      code: "custom",
      path: ["judge_model"],
      message: t("guardrails.llmJudge.judgeModel.required"),
    });
  }
  const criteria = Array.isArray(values.criteria) ? values.criteria : [];
  criteria.forEach((criterion, index) => {
    const item = (criterion ?? {}) as Record<string, unknown>;
    if (isBlankGuardrailValue(item.name)) {
      ctx.addIssue({
        code: "custom",
        path: ["criteria", index, "name"],
        message: t("guardrails.llmJudge.criteria.nameRequired"),
      });
    }
    if (isBlankGuardrailValue(item.weight)) {
      ctx.addIssue({
        code: "custom",
        path: ["criteria", index, "weight"],
        message: t("guardrails.llmJudge.criteria.weightRequired"),
      });
    }
    if (isBlankGuardrailValue(item.description)) {
      ctx.addIssue({
        code: "custom",
        path: ["criteria", index, "description"],
        message: t("guardrails.llmJudge.criteria.descriptionRequired"),
      });
    }
  });
};

const validateStepZero = (
  values: Record<string, unknown>,
  options: AddGuardrailSchemaOptions,
  ctx: z.RefinementCtx,
) => {
  const { selectedProvider, providerParams, isDecisionModelProvider, isToolPermissionProvider, t } = options;
  if (isBlankGuardrailValue(values.guardrail_name)) {
    ctx.addIssue({
      code: "custom",
      path: ["guardrail_name"],
      message: t("guardrails.addForm.fields.name.required"),
    });
  }
  if (isBlankGuardrailValue(values.provider)) {
    ctx.addIssue({
      code: "custom",
      path: ["provider"],
      message: t("guardrails.addForm.fields.provider.required"),
    });
  }
  if (isBlankGuardrailValue(values.mode)) {
    ctx.addIssue({
      code: "custom",
      path: ["mode"],
      message: t("guardrails.addForm.fields.mode.required"),
    });
  }

  if (shouldRenderLLMJudgeFields(selectedProvider)) {
    validateLlmJudgeFields(values, t, ctx);
  }

  const usesOwnConfigurationFields = [
    isToolPermissionProvider,
    shouldRenderContentFilterConfigSettings(selectedProvider),
    shouldRenderLLMJudgeFields(selectedProvider),
    isDecisionModelProvider,
  ].some(Boolean);

  if (!usesOwnConfigurationFields && selectedProvider && providerParams) {
    const providerKey = guardrail_provider_map[selectedProvider]?.toLowerCase();
    const providerFields = providerKey ? providerParams[providerKey] : undefined;
    const providerValidation = { values, selectedProvider, providerFields, t, ctx };
    validateProviderFields(providerValidation);
  }
};

const validateStepOne = (values: Record<string, unknown>, options: AddGuardrailSchemaOptions, ctx: z.RefinementCtx) => {
  const { selectedProvider, providerParams, isDecisionModelProvider, isToolPermissionProvider, t } = options;
  if (isDecisionModelProvider && isBlankGuardrailValue(values.decision_model)) {
    ctx.addIssue({
      code: "custom",
      path: ["decision_model"],
      message: t("guardrails.decisionModel.model.required"),
    });
  }

  const isCustomProvider = isDecisionModelProvider || isToolPermissionProvider;
  if (!isCustomProvider && selectedProvider && providerParams) {
    const providerKey = guardrail_provider_map[selectedProvider]?.toLowerCase();
    const providerFields = providerKey ? providerParams[providerKey] : undefined;
    if (providerFields?.optional_params && asText(values.provider) === selectedProvider) {
      validateOptionalParams(values, providerFields, t, ctx);
    }
  }
};

export const createAddGuardrailSchema = (options: AddGuardrailSchemaOptions) =>
  z.looseObject({}).superRefine((values, ctx) => {
    if (options.currentStep === 0) {
      validateStepZero(values, options, ctx);
      return;
    }
    validateStepOne(values, options, ctx);
  });
