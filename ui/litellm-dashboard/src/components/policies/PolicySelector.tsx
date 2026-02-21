import React, { useEffect, useState } from "react";
import { Select } from "antd";
import { Policy } from "./types";
import { getPoliciesList } from "../networking";

interface PolicySelectorProps {
  onChange: (selectedPolicies: string[]) => void;
  value?: string[];
  className?: string;
  accessToken: string;
  disabled?: boolean;
}

const PolicySelector: React.FC<PolicySelectorProps> = ({ 
  onChange, 
  value, 
  className, 
  accessToken, 
  disabled 
}) => {
  const [policies, setPolicies] = useState<Policy[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const fetchPolicies = async () => {
      if (!accessToken) return;

      setLoading(true);
      try {
        const response = await getPoliciesList(accessToken);
        console.log("Policies response:", response);
        if (response.policies) {
          console.log("Policies data:", response.policies);
          setPolicies(response.policies);
        }
      } catch (error) {
        console.error("Error fetching policies:", error);
      } finally {
        setLoading(false);
      }
    };

    fetchPolicies();
  }, [accessToken]);

  const handlePolicyChange = (selectedValues: string[]) => {
    console.log("Selected policies:", selectedValues);
    onChange(selectedValues);
  };

  return (
    <div>
      <Select
        mode="multiple"
        disabled={disabled}
        placeholder={disabled ? "Setting policies is a premium feature." : "Select policies"}
        onChange={handlePolicyChange}
        value={value}
        loading={loading}
        className={className}
        allowClear
        options={policies.map((policy) => {
          console.log("Mapping policy:", policy);
          return {
            label: `${policy.policy_name}${policy.description ? ` - ${policy.description}` : ""}`,
            value: policy.policy_name,
          };
        })}
        optionFilterProp="label"
        showSearch
        style={{ width: "100%" }}
      />
    </div>
  );
};

export default PolicySelector;
