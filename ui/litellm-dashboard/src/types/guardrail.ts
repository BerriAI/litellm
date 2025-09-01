export interface Guardrail {
  guardrail_id: string;
  guardrail_name: string | null;
  litellm_params: {
    guardrail: string;
    mode: string;
    default_on: boolean;
    [key: string]: any;
  };
  guardrail_info?: Record<string, any>;
}
