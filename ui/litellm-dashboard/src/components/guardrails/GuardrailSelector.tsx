import React, { useEffect, useState } from "react";
import { Select } from "antd";
import { Guardrail } from "./types";
import { getGuardrailsList } from "../networking";

interface GuardrailSelectorProps {
  onChange: (selectedGuardrails: string[]) => void;
  value?: string[];
  className?: string;
  accessToken: string;
  disabled?: boolean;
}

const GuardrailSelector: React.FC<GuardrailSelectorProps> = ({ onChange, value, className, accessToken, disabled }) => {
  const [guardrails, setGuardrails] = useState<Guardrail[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const fetchGuardrails = async () => {
      if (!accessToken) return;

      setLoading(true);
      try {
        const response = await getGuardrailsList(accessToken);
        console.log("Guardrails response:", response);
        if (response.guardrails) {
          console.log("Guardrails data:", response.guardrails);
          setGuardrails(response.guardrails);
        }
      } catch (error) {
        console.error("Error fetching guardrails:", error);
      } finally {
        setLoading(false);
      }
    };

    fetchGuardrails();
  }, [accessToken]);

  const handleGuardrailChange = (selectedValues: string[]) => {
    console.log("Selected guardrails:", selectedValues);
    onChange(selectedValues);
  };

  return (
    <div>
      <Select
        mode="multiple"
        disabled={disabled}
        placeholder={disabled ? "Setting guardrails is a premium feature." : "Select guardrails"}
        onChange={handleGuardrailChange}
        value={value}
        loading={loading}
        className={className}
        options={guardrails.map((guardrail) => {
          console.log("Mapping guardrail:", guardrail);
          return {
            label: `${guardrail.guardrail_name}`,
            value: guardrail.guardrail_name,
          };
        })}
        optionFilterProp="label"
        showSearch
        style={{ width: "100%" }}
      />
    </div>
  );
};

export default GuardrailSelector;
