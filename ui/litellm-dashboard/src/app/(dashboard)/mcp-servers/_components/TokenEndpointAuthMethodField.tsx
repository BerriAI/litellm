import type { ParseKeys } from "i18next";
import { Info } from "lucide-react";
import React from "react";
import { useTranslation } from "react-i18next";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { SimpleTooltip } from "@/components/ui/tooltip";

import { MountedFormField } from "@/components/common_components/MountedFormField";
import { selectControl, selectTriggerControl } from "./mcpFieldRules";

const TOKEN_ENDPOINT_AUTH_METHOD_OPTIONS: ReadonlyArray<{ value: string; labelKey: ParseKeys }> = [
  { value: "client_secret_basic", labelKey: "mcpServers.tokenEndpointAuthMethodField.methods.clientSecretBasic" },
  { value: "client_secret_post", labelKey: "mcpServers.tokenEndpointAuthMethodField.methods.clientSecretPost" },
];

interface TokenEndpointAuthMethodFieldProps {
  isEditing?: boolean;
}

const TokenEndpointAuthMethodField: React.FC<TokenEndpointAuthMethodFieldProps> = ({ isEditing = false }) => {
  const { t } = useTranslation();
  const authMethodOptions = TOKEN_ENDPOINT_AUTH_METHOD_OPTIONS.map((option) => ({
    value: option.value,
    label: t(option.labelKey),
  }));
  return (
    <MountedFormField
      label={
        <span className="text-sm font-medium text-foreground flex items-center">
          {t("mcpServers.tokenEndpointAuthMethodField.label")}
          <SimpleTooltip content={t("mcpServers.tokenEndpointAuthMethodField.tooltip")}>
            <Info className="ml-2 size-4 text-info hover:text-info/80 cursor-help" />
          </SimpleTooltip>
        </span>
      }
      name={["credentials", "token_endpoint_auth_method"]}
    >
      {(control) => {
        const placeholder = isEditing
          ? t("mcpServers.tokenEndpointAuthMethodField.placeholderEditing")
          : t("mcpServers.tokenEndpointAuthMethodField.placeholder");
        return (
          <Select {...selectControl<string>(control)} items={authMethodOptions}>
            <SelectTrigger {...selectTriggerControl(control)} className="w-full rounded-lg">
              <SelectValue placeholder={placeholder} />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value={null}>{placeholder}</SelectItem>
              {authMethodOptions.map((option) => (
                <SelectItem key={option.value} value={option.value}>
                  {option.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        );
      }}
    </MountedFormField>
  );
};

export default TokenEndpointAuthMethodField;
