export type CallbackMode = "success" | "failure" | "success_and_failure";

export interface AlertingObject {
  name: string;
  variables: Record<string, string | null>;
  type?: CallbackMode;
}

export interface AlertingVariables {
  SLACK_WEBHOOK_URL: string | null;
  LANGFUSE_PUBLIC_KEY: string | null;
  LANGFUSE_SECRET_KEY: string | null;
  LANGFUSE_HOST: string | null;
  OPENMETER_API_KEY: string | null;
}

export interface AvailableCallback {
  litellm_callback_name: string;
  litellm_callback_params: string[];
  ui_callback_name: string;
}

export interface CallbackConfigParam {
  type?: string;
  ui_name?: string;
  required?: boolean;
}

export interface CallbackConfig {
  id: string;
  displayName: string;
  logo?: string;
  dynamic_params?: Record<string, CallbackConfigParam>;
}

export interface AlertData {
  name: string;
  variables: { SLACK_WEBHOOK_URL: string };
  active_alerts: string[];
  alerts_to_webhook: Record<string, string>;
}
