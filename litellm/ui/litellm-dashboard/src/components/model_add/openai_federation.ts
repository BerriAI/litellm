import i18n from "@/i18n/i18n";
import { isBlank, type FederationField } from "./federation_field";

const OPENAI_API_HOST = "api.openai.com";

export const OPENAI_FEDERATION_FIELDS: readonly FederationField[] = [
  {
    key: "openai_identity_provider_id",
    labelKey: "modelAdd.openaiFederation.fields.identityProviderId.label",
    tooltipKey: "modelAdd.openaiFederation.fields.identityProviderId.tooltip",
    required: false,
    control: "text",
  },
  {
    key: "openai_service_account_id",
    labelKey: "modelAdd.openaiFederation.fields.serviceAccountId.label",
    tooltipKey: "modelAdd.openaiFederation.fields.serviceAccountId.tooltip",
    required: false,
    control: "text",
  },
  {
    key: "openai_identity_token_file",
    labelKey: "modelAdd.openaiFederation.fields.identityTokenFile.label",
    tooltipKey: "modelAdd.openaiFederation.fields.identityTokenFile.tooltip",
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
    : i18n.t("modelAdd.openaiFederation.apiBaseNotOpenAI", { host: OPENAI_API_HOST });
