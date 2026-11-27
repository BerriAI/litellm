import type { TFunction } from "i18next";
import { z } from "zod";
import { guardrail_provider_map } from "./guardrail_info_helpers";
import {
  isBlankGuardrailValue,
  validateOptionalParams,
  validateProviderFields,
  type GuardrailProviderParamsMap,
} from "./guardrailFormValidation";

interface GuardrailInfoSchemaOptions {
  guardrailProvider: string | null;
  providerParams: GuardrailProviderParamsMap | null;
  t: TFunction;
}

export const createGuardrailInfoSchema = ({ guardrailProvider, providerParams, t }: GuardrailInfoSchemaOptions) =>
  z.looseObject({}).superRefine((values, ctx) => {
    if (isBlankGuardrailValue(values.guardrail_name)) {
      ctx.addIssue({
        code: "custom",
        path: ["guardrail_name"],
        message: t("guardrails.info.settings.nameRequired"),
      });
    }

    if (!guardrailProvider || guardrailProvider === "tool_permission" || !providerParams) {
      return;
    }

    const currentProvider =
      Object.keys(guardrail_provider_map).find((key) => guardrail_provider_map[key] === guardrailProvider) ?? null;
    if (!currentProvider) return;

    const providerKey = guardrail_provider_map[currentProvider]?.toLowerCase();
    const providerFields = providerKey ? providerParams[providerKey] : undefined;
    const providerValidation = {
      values,
      selectedProvider: currentProvider,
      providerFields,
      t,
      ctx,
    };
    validateProviderFields(providerValidation);
    validateOptionalParams(values, providerFields, t, ctx);
  });
