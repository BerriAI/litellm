import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { GuardrailField, type GuardrailFormControl } from "./GuardrailFormField";
import { hasMCPBlockMode, mcpBlockAction, type MCPBlockAction } from "./guardrail_info_helpers";

const OPTIONS = [
  { value: "tool_error", label: "Return a tool error and let the model continue" },
  { value: "raise", label: "Stop the request with an error" },
];

export const MCPBlockPolicySelect = ({
  id,
  value,
  onChange,
}: {
  id: string;
  value: MCPBlockAction;
  onChange: (value: MCPBlockAction) => void;
}) => (
  <div className="space-y-2">
    <Select items={OPTIONS} value={value} onValueChange={(next) => next !== null && onChange(mcpBlockAction(next))}>
      <SelectTrigger id={id} aria-describedby={`${id}-help`} className="w-full">
        <SelectValue />
      </SelectTrigger>
      <SelectContent>
        {OPTIONS.map((option) => (
          <SelectItem key={option.value} value={option.value}>
            {option.label}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
    <p id={`${id}-help`} className="text-sm text-muted-foreground">
      Applies to pre_mcp_call and during_mcp_call blocks. Stop returns HTTP 400 for non-streaming Chat Completions and
      Responses requests whose tools LiteLLM executes. Streaming and native MCP do not support Stop. A during-call block
      cannot undo a tool that has already run.
    </p>
  </div>
);

export const MCPBlockPolicyField = ({ control, mode }: { control: GuardrailFormControl; mode: unknown }) => {
  if (!hasMCPBlockMode(mode)) return null;
  return (
    <GuardrailField control={control} name="on_mcp_block" label="When an MCP tool is blocked">
      {({ id, value, onChange }) => <MCPBlockPolicySelect id={id} value={mcpBlockAction(value)} onChange={onChange} />}
    </GuardrailField>
  );
};

export const MCPBlockPolicyDetail = ({ mode, value }: { mode: unknown; value: unknown }) => {
  if (!hasMCPBlockMode(mode)) return null;
  return (
    <div>
      <p className="font-medium">When an MCP tool is blocked</p>
      <div>{OPTIONS.find((option) => option.value === mcpBlockAction(value))?.label}</div>
    </div>
  );
};
