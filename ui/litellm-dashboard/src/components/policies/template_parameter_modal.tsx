import React, { useState, useEffect } from "react";
import { Modal, Spin } from "antd";
import { Button, TextInput } from "@tremor/react";

interface TemplateParameter {
  name: string;
  label: string;
  type: string;
  required: boolean;
  placeholder?: string;
}

interface TemplateParameterModalProps {
  visible: boolean;
  template: any;
  onConfirm: (parameters: Record<string, string>) => void;
  onCancel: () => void;
  isLoading?: boolean;
}

const TemplateParameterModal: React.FC<TemplateParameterModalProps> = ({
  visible,
  template,
  onConfirm,
  onCancel,
  isLoading = false,
}) => {
  const [parameterValues, setParameterValues] = useState<Record<string, string>>({});

  const parameters: TemplateParameter[] = template?.parameters || [];

  useEffect(() => {
    if (visible && template) {
      const initial: Record<string, string> = {};
      parameters.forEach((p) => {
        initial[p.name] = "";
      });
      setParameterValues(initial);
    }
  }, [visible, template]);

  const allRequiredFilled = parameters
    .filter((p) => p.required)
    .every((p) => (parameterValues[p.name] || "").trim().length > 0);

  const handleConfirm = () => {
    onConfirm(parameterValues);
  };

  return (
    <Modal
      title={
        <div>
          <h3 className="text-lg font-semibold mb-1">{template?.title}</h3>
          <p className="text-sm text-gray-500 font-normal">
            {template?.llm_enrichment
              ? "Enter your brand name to auto-discover competitors and configure guardrails"
              : "Configure template parameters"}
          </p>
        </div>
      }
      open={visible}
      onCancel={onCancel}
      width={500}
      footer={[
        <Button key="cancel" variant="secondary" onClick={onCancel} disabled={isLoading}>
          Cancel
        </Button>,
        <Button
          key="confirm"
          onClick={handleConfirm}
          loading={isLoading}
          disabled={!allRequiredFilled || isLoading}
        >
          {isLoading
            ? template?.llm_enrichment
              ? "Discovering competitors..."
              : "Processing..."
            : "Continue"}
        </Button>,
      ]}
    >
      <div className="py-4 space-y-4">
        {parameters.map((param) => (
          <div key={param.name}>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              {param.label}
              {param.required && <span className="text-red-500 ml-1">*</span>}
            </label>
            <TextInput
              placeholder={param.placeholder || ""}
              value={parameterValues[param.name] || ""}
              onChange={(e) =>
                setParameterValues((prev) => ({
                  ...prev,
                  [param.name]: e.target.value,
                }))
              }
            />
          </div>
        ))}

        {template?.llm_enrichment && (
          <div className="mt-4 p-3 bg-blue-50 rounded-lg border border-blue-100">
            <p className="text-sm text-blue-800">
              This template uses AI to automatically discover your competitors and configure
              guardrails. An onboarded LLM will be called to identify competitor names.
            </p>
          </div>
        )}

        {isLoading && (
          <div className="flex items-center gap-3 mt-4 p-3 bg-gray-50 rounded-lg">
            <Spin size="small" />
            <span className="text-sm text-gray-600">
              {template?.llm_enrichment
                ? "Using AI to discover competitors..."
                : "Processing template..."}
            </span>
          </div>
        )}
      </div>
    </Modal>
  );
};

export default TemplateParameterModal;
