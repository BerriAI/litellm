import { Info } from "lucide-react";
import React from "react";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { SimpleTooltip } from "@/components/ui/tooltip";

import { MountedFormField } from "@/components/common_components/MountedFormField";
import { selectControl, selectTriggerControl } from "./mcpFieldRules";

const TOKEN_ENDPOINT_AUTH_METHOD_OPTIONS = [
  { value: "client_secret_basic", label: "Client Secret Basic" },
  { value: "client_secret_post", label: "Client Secret Post" },
];

interface TokenEndpointAuthMethodFieldProps {
  isEditing?: boolean;
}

const TokenEndpointAuthMethodField: React.FC<TokenEndpointAuthMethodFieldProps> = ({ isEditing = false }) => (
  <MountedFormField
    label={
      <span className="text-sm font-medium text-foreground flex items-center">
        Token Endpoint Auth Method (optional)
        <SimpleTooltip content="How the proxy authenticates to the upstream OAuth token endpoint. Client Secret Basic sends the client credentials in an HTTP Basic Authorization header; leave blank to use the default, Client Secret Post, which sends them in the request body.">
          <Info className="ml-2 size-4 text-info hover:text-info/80 cursor-help" />
        </SimpleTooltip>
      </span>
    }
    name={["credentials", "token_endpoint_auth_method"]}
  >
    {(control) => {
      const placeholder = isEditing
        ? "Leave blank to keep existing (default Client Secret Post)"
        : "Default (Client Secret Post)";
      return (
        <Select {...selectControl<string>(control)} items={TOKEN_ENDPOINT_AUTH_METHOD_OPTIONS}>
          <SelectTrigger {...selectTriggerControl(control)} className="w-full rounded-lg">
            <SelectValue placeholder={placeholder} />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value={null}>{placeholder}</SelectItem>
            {TOKEN_ENDPOINT_AUTH_METHOD_OPTIONS.map((option) => (
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

export default TokenEndpointAuthMethodField;
