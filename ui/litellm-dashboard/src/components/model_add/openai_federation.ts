import { isBlank, type FederationField } from "./federation_field";

const OPENAI_API_HOST = "api.openai.com";

export const OPENAI_FEDERATION_FIELDS: readonly FederationField[] = [
  {
    key: "openai_identity_provider_id",
    label: "Identity Provider ID",
    tooltip:
      "The workload identity provider registered in your OpenAI organization that trusts the token's issuer. Leave empty when the proxy sets OPENAI_IDENTITY_PROVIDER_ID in its environment.",
    required: false,
    control: "text",
  },
  {
    key: "openai_service_account_id",
    label: "Service Account ID",
    tooltip:
      "The OpenAI service account the verified identity acts as. Its project decides what the model can reach. Leave empty when the proxy sets OPENAI_SERVICE_ACCOUNT_ID in its environment.",
    required: false,
    control: "text",
  },
  {
    key: "openai_identity_token_file",
    label: "Identity Token File",
    tooltip:
      "Absolute path on the proxy host of a file holding the identity token, for example a projected Kubernetes service account token. Leave empty when the proxy sets OPENAI_IDENTITY_TOKEN_FILE in its environment.",
    placeholder: "/var/run/secrets/kubernetes.io/serviceaccount/token",
    required: false,
    control: "text",
  },
];

export const isOpenAIProvider = (provider: string | null | undefined): boolean => provider?.toLowerCase() === "openai";

const isOpenAIApiHost = (hostname: string): boolean =>
  hostname === OPENAI_API_HOST || hostname.endsWith(`.${OPENAI_API_HOST}`);

const targetsOpenAIApi = (apiBase: string): boolean => {
  const url = URL.canParse(apiBase) ? new URL(apiBase) : null;
  return url?.protocol === "https:" && isOpenAIApiHost(url.hostname);
};

export const validateOpenAIFederationApiBase = (value: unknown): string | true =>
  typeof value !== "string" || isBlank(value) || targetsOpenAIApi(value)
    ? true
    : `Workload identity federation only reaches the OpenAI API. Enter https://${OPENAI_API_HOST}/v1 or a regional *.${OPENAI_API_HOST} URL, or leave this empty`;
