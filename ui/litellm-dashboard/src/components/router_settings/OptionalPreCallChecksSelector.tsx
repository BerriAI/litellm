import React from "react";
import { Select } from "antd";

export const OPTIONAL_PRE_CALL_CHECK_OPTIONS = [
  "prompt_caching",
  "router_budget_limiting",
  "responses_api_deployment_check",
  "deployment_affinity",
  "session_affinity",
  "enforce_model_rate_limits",
  "encrypted_content_affinity",
] as const;

interface OptionalPreCallChecksSelectorProps {
  value: string[];
  onChange: (value: string[]) => void;
}

const OptionalPreCallChecksSelector: React.FC<OptionalPreCallChecksSelectorProps> = ({ value, onChange }) => {
  return (
    <div className="space-y-2 max-w-3xl">
      <label className="block">
        <span className="text-xs font-medium text-gray-700 uppercase tracking-wide">Optional Pre-call Checks</span>
        <p className="text-xs text-gray-500 mt-0.5 mb-2">
          Extra checks the router runs before picking a deployment. Add &apos;prompt_caching&apos; to route repeat
          requests back to the deployment that cached the prompt.{" "}
          <a
            href="https://docs.litellm.ai/docs/tutorials/claude_code_prompt_cache_routing"
            target="_blank"
            rel="noopener noreferrer"
            className="text-blue-600 hover:text-blue-800 underline"
          >
            Learn more
          </a>
        </p>
        <Select
          mode="multiple"
          value={value}
          onChange={onChange}
          options={OPTIONAL_PRE_CALL_CHECK_OPTIONS.map((option) => ({ value: option, label: option }))}
          placeholder="No pre-call checks enabled"
          className="w-full"
          data-testid="optional-pre-call-checks-select"
        />
      </label>
    </div>
  );
};

export default OptionalPreCallChecksSelector;
