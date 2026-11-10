import type { TFunction } from "i18next";
import i18n from "@/i18n/i18n";
import { AUTH_TYPE, TRANSPORT } from "@/components/mcp_tools/types";
import { AUTH_TYPES_REQUIRING_AUTH_VALUE, hasUnfinishedHeaderRow, reduceStaticHeaders } from "./createServerPayload";

const previewCredentials = (
  authValue: unknown,
  needsAuthValue: boolean,
  secretHeaders: Readonly<Record<string, string>>,
) => {
  if (needsAuthValue && typeof authValue === "string" && authValue.trim()) return { auth_value: authValue };
  if (Object.keys(secretHeaders).length > 0) return { headers: secretHeaders };
  return undefined;
};

const connectionConfig = (values: Readonly<Record<string, unknown>>) => {
  const credentials = values.credentials;
  const authValue =
    credentials && typeof credentials === "object" && "auth_value" in credentials ? credentials.auth_value : undefined;
  const needsAuthValue =
    typeof values.auth_type === "string" && AUTH_TYPES_REQUIRING_AUTH_VALUE.includes(values.auth_type);
  const secretHeaders =
    values.auth_type === AUTH_TYPE.SECRET_HEADERS
      ? Object.fromEntries(
          Object.entries(reduceStaticHeaders(values.secret_headers)).sort(([a], [b]) => a.localeCompare(b)),
        )
      : {};
  return {
    url: typeof values.url === "string" ? values.url : "",
    transport: typeof values.transport === "string" ? values.transport : "",
    auth_type: typeof values.auth_type === "string" ? values.auth_type : "",
    static_headers: Object.fromEntries(
      Object.entries(reduceStaticHeaders(values.static_headers)).sort(([a], [b]) => a.localeCompare(b)),
    ),
    credentials: previewCredentials(authValue, needsAuthValue, secretHeaders),
  };
};

type EditToolPreview =
  | { readonly kind: "saved" }
  | { readonly kind: "incomplete"; readonly message?: string }
  | { readonly kind: "preview"; readonly config: ReturnType<typeof connectionConfig> };

export const getEditToolPreview = (
  values: Readonly<Record<string, unknown>>,
  initialValues: Readonly<Record<string, unknown>>,
  t: TFunction = i18n.t.bind(i18n),
): EditToolPreview => {
  const staticAuth =
    values.auth_type === AUTH_TYPE.NONE ||
    values.auth_type === AUTH_TYPE.SECRET_HEADERS ||
    (typeof values.auth_type === "string" && AUTH_TYPES_REQUIRING_AUTH_VALUE.includes(values.auth_type));
  if (!staticAuth || ![TRANSPORT.HTTP, TRANSPORT.SSE].includes(String(values.transport))) {
    return { kind: "saved" };
  }

  const config = connectionConfig(values);
  if (JSON.stringify(config) === JSON.stringify(connectionConfig(initialValues))) {
    return { kind: "saved" };
  }

  const missingNewCredential =
    config.auth_type !== initialValues.auth_type &&
    (AUTH_TYPES_REQUIRING_AUTH_VALUE.includes(config.auth_type) || config.auth_type === AUTH_TYPE.SECRET_HEADERS) &&
    config.credentials === undefined;
  const validUrl = URL.canParse(config.url) && ["http:", "https:"].includes(new URL(config.url).protocol);
  const incompleteHeaders = [
    ...Object.values(config.static_headers),
    ...Object.values(config.credentials?.headers ?? {}),
  ].some((value) => !value.trim());
  const unfinishedSecretRow =
    config.auth_type === AUTH_TYPE.SECRET_HEADERS && hasUnfinishedHeaderRow(values.secret_headers);
  if (!validUrl || missingNewCredential || incompleteHeaders || unfinishedSecretRow) {
    return { kind: "incomplete" };
  }
  const savedConfig = connectionConfig(initialValues);
  const changedOrigin =
    !URL.canParse(savedConfig.url) || new URL(config.url).origin !== new URL(savedConfig.url).origin;
  const reusesHeader = Object.entries(config.static_headers).some(
    ([key, value]) => savedConfig.static_headers[key] === value,
  );
  const needsSavedCredential =
    (AUTH_TYPES_REQUIRING_AUTH_VALUE.includes(config.auth_type) || config.auth_type === AUTH_TYPE.SECRET_HEADERS) &&
    !config.credentials;
  if (changedOrigin && (needsSavedCredential || reusesHeader)) {
    return {
      kind: "incomplete",
      message: t("mcpServers.editToolPreview.originChanged"),
    };
  }
  return { kind: "preview", config };
};
