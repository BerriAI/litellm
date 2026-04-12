import React, { useEffect, useState } from "react";
import { Select } from "antd";
import { Policy } from "./types";
import { getPoliciesList } from "../networking";

/** Prefix for policy version IDs in request body; must match backend POLICY_VERSION_ID_PREFIX. */
export const POLICY_VERSION_ID_PREFIX = "policy_";

/** Build the value sent in the request body: policy_<uuid> so backend executes this exact version. */
export function policyVersionRef(policyId: string): string {
  return `${POLICY_VERSION_ID_PREFIX}${policyId}`;
}

/** Build select options from policies (filter non-draft, label with name/version/status). */
export function getPolicyOptionEntries(policies: Policy[]): { value: string; label: string }[] {
  return policies
    .filter((policy) => (policy.version_status ?? "draft") !== "draft")
    .map((policy) => {
      const versionNum = policy.version_number ?? 1;
      const status = policy.version_status ?? "draft";
      const label = `${policy.policy_name} — v${versionNum} (${status})${
        policy.description ? ` — ${policy.description}` : ""
      }`;
      const isProduction = status === "production";
      return {
        label,
        value: isProduction
          ? policy.policy_name
          : policy.policy_id
            ? policyVersionRef(policy.policy_id)
            : policy.policy_name,
      };
    });
}

interface PolicySelectorProps {
  onChange: (selectedPolicies: string[]) => void;
  value?: string[];
  className?: string;
  accessToken: string;
  disabled?: boolean;
  /** Called after policies are loaded; use to build value→label map for display elsewhere. */
  onPoliciesLoaded?: (policies: Policy[]) => void;
}

const PolicySelector: React.FC<PolicySelectorProps> = ({
  onChange,
  value,
  className,
  accessToken,
  disabled,
  onPoliciesLoaded,
}) => {
  const [policies, setPolicies] = useState<Policy[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const fetchPolicies = async () => {
      if (!accessToken) return;

      setLoading(true);
      try {
        const response = await getPoliciesList(accessToken);
        if (response.policies) {
          setPolicies(response.policies);
          onPoliciesLoaded?.(response.policies);
        }
      } catch (error) {
        console.error("Error fetching policies:", error);
      } finally {
        setLoading(false);
      }
    };

    fetchPolicies();
  }, [accessToken, onPoliciesLoaded]);

  const handlePolicyChange = (selectedValues: string[]) => {
    onChange(selectedValues);
  };

  return (
    <div>
      <Select
        mode="multiple"
        disabled={disabled}
        placeholder={
          disabled
            ? "Setting policies is a premium feature."
            : "Select policies (production or published versions)"
        }
        onChange={handlePolicyChange}
        value={value}
        loading={loading}
        className={className}
        allowClear
        options={getPolicyOptionEntries(policies)}
        optionFilterProp="label"
        showSearch
        style={{ width: "100%" }}
      />
    </div>
  );
};

export default PolicySelector;
