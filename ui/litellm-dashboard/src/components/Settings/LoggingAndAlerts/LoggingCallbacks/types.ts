export type CallbackMode = "success" | "failure" | "success_and_failure";

export interface AlertingObject {
  name: string;
  variables: AlertingVariables;
  type?: CallbackMode;
}

export interface AlertingVariables {
  SLACK_WEBHOOK_URL: string | null;
  LANGFUSE_PUBLIC_KEY: string | null;
  LANGFUSE_SECRET_KEY: string | null;
  LANGFUSE_HOST: string | null;
  OPENMETER_API_KEY: string | null;
}
