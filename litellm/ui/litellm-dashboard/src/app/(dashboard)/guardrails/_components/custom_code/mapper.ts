import {
  loggingOnlyContinueFromParams,
  loggingOnlyScopeToChoice,
  streamScopeByModeFromConfig,
  streamScopeForUpdate,
  streamScopePayload,
} from "../guardrail_info_helpers";
import type { LoggingOnlyScope } from "../guardrail_info_helpers";
import { getCustomCodeLoggingOnlyScopeCreate, getCustomCodeLoggingOnlyScopeUpdate } from "./CustomCodeLoggingOnlyScope";
import { CODE_TEMPLATES } from "./custom_code_catalog";
import type { CustomCodeFormValues } from "./schema";

export interface EditGuardrailData {
  guardrail_id: string;
  guardrail_name: string;
  litellm_params: {
    mode?: string | string[] | Record<string, unknown>;
    default_on?: boolean;
    custom_code?: string;
    logging_only_scope?: LoggingOnlyScope | null;
    [key: string]: any;
  };
}

const normalizeCustomCodeMode = (m: string | string[] | Record<string, unknown> | undefined): string[] => {
  if (m === undefined || m === null) return ["pre_call"];
  if (Array.isArray(m)) return m.length ? m : ["pre_call"];
  if (typeof m === "string") return [m];
  return [];
};

export const customCodeToFormValues = (editData?: EditGuardrailData | null): CustomCodeFormValues => {
  if (!editData) {
    return {
      guardrail_name: "",
      mode: ["pre_call"],
      stream_scope_by_mode: {},
      logging_only_scope_choice: "default",
      logging_only_continue: false,
      default_on: false,
      code: CODE_TEMPLATES.empty.code,
    };
  }
  const normalizedMode = normalizeCustomCodeMode(editData.litellm_params?.mode);
  return {
    guardrail_name: editData.guardrail_name || "",
    mode: normalizedMode,
    stream_scope_by_mode: streamScopeByModeFromConfig(editData.litellm_params?.stream_scope, normalizedMode),
    logging_only_scope_choice: loggingOnlyScopeToChoice(editData.litellm_params?.logging_only_scope),
    logging_only_continue: loggingOnlyContinueFromParams(editData.litellm_params),
    default_on: editData.litellm_params?.default_on || false,
    code: editData.litellm_params?.custom_code || CODE_TEMPLATES.empty.code,
  };
};

export const buildCustomCodeCreatePayload = (values: CustomCodeFormValues) => {
  const streamScope = streamScopePayload(values.mode, values.stream_scope_by_mode);
  return {
    guardrail_name: values.guardrail_name,
    litellm_params: {
      guardrail: "custom_code",
      mode: values.mode,
      default_on: values.default_on,
      custom_code: values.code,
      ...(streamScope !== undefined ? { stream_scope: streamScope } : {}),
      ...getCustomCodeLoggingOnlyScopeCreate(
        values.mode,
        values.logging_only_scope_choice,
        values.logging_only_continue,
      ),
    },
    guardrail_info: {},
  };
};

export const buildCustomCodeUpdatePayload = (values: CustomCodeFormValues, editData: EditGuardrailData) => {
  const updateData: Record<string, any> = {
    litellm_params: {
      custom_code: values.code,
      ...getCustomCodeLoggingOnlyScopeUpdate(
        values.mode,
        editData.litellm_params,
        values.logging_only_scope_choice,
        values.logging_only_continue,
      ),
    },
  };

  if (values.guardrail_name !== editData.guardrail_name) {
    updateData.guardrail_name = values.guardrail_name;
  }
  const existingMode = normalizeCustomCodeMode(editData.litellm_params?.mode);
  const modeChanged = values.mode.length !== existingMode.length || values.mode.some((m, i) => m !== existingMode[i]);
  if (modeChanged) {
    updateData.litellm_params.mode = values.mode;
  }
  if (values.default_on !== editData.litellm_params?.default_on) {
    updateData.litellm_params.default_on = values.default_on;
  }
  const nextStreamScope = streamScopeForUpdate(
    values.mode,
    values.stream_scope_by_mode,
    editData.litellm_params?.stream_scope,
    existingMode,
  );
  if (nextStreamScope !== undefined) {
    updateData.litellm_params.stream_scope = nextStreamScope;
  }

  return updateData;
};

export const deriveCustomCodeTestInputType = (mode: string[]): "request" | "response" => {
  const requestModes = ["pre_call", "pre_mcp_call"];
  const responseModes = ["post_call", "post_mcp_call"];
  if (mode.some((m) => requestModes.includes(m))) {
    return "request";
  }
  if (mode.some((m) => responseModes.includes(m))) {
    return "response";
  }
  return "request";
};
