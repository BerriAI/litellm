import React, { useEffect, useState } from "react";
import { Guardrail } from "./types";
import { getGuardrailsList } from "../networking";
import { MultiSelect } from "@/components/shared/MultiSelect";

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
        if (response.guardrails) {
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
    onChange(selectedValues);
  };

  return (
    <div className="min-w-0">
      <MultiSelect
        disabled={disabled}
        placeholder={disabled ? "Setting guardrails is a premium feature." : "Select guardrails"}
        onValueChange={handleGuardrailChange}
        value={value}
        loading={loading}
        className={className}
        options={guardrails.flatMap((guardrail) => {
          const name = guardrail.guardrail_name;
          if (name == null || name === "") {
            return [];
          }
          return [{ label: name, value: name }];
        })}
      />
    </div>
  );
};

export default GuardrailSelector;
