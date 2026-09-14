import { Info } from "lucide-react";
import React from "react";
import { SimpleTooltip } from "@/components/ui/tooltip";
import { Input } from "@/components/ui/input";

import { MountedFormField } from "@/components/common_components/MountedFormField";
import { textControl } from "./mcpFieldRules";

const UpstreamTokenHeaderField: React.FC = () => (
  <MountedFormField
    label={
      <span className="text-sm font-medium text-foreground flex items-center">
        Token Header (optional)
        <SimpleTooltip content="Which upstream header carries the token LiteLLM resolves for this server. Leave blank to send it as 'Authorization: Bearer <token>', which is the default and what most servers expect. Set a header name when the upstream expects it elsewhere, for example an API gateway that terminates its own credential on 'esb-oauth' while a separate Authorization from Static Headers passes through to the server behind it.">
          <Info className="ml-2 size-4 text-info hover:text-info/80 cursor-help" />
        </SimpleTooltip>
      </span>
    }
    name={["credentials", "upstream_token_header"]}
  >
    {(control) => (
      <Input
        {...textControl(control)}
        placeholder="Authorization"
        className="rounded-lg border-border focus:border-info focus:ring-ring"
      />
    )}
  </MountedFormField>
);

export default UpstreamTokenHeaderField;
