import React, { useState, useEffect } from "react";
import { Card, Title, Subtitle } from "@tremor/react";
import { Form, Input, Select, Tooltip, Alert } from "antd";
import { InfoCircleOutlined, PlusOutlined, DeleteOutlined } from "@ant-design/icons";
import GuardrailSelector from "../guardrails/GuardrailSelector";

interface PassThroughGuardrailsSectionProps {
  accessToken: string;
  value?: Record<string, { request_fields?: string[]; response_fields?: string[] } | null>;
  onChange?: (guardrails: Record<string, { request_fields?: string[]; response_fields?: string[] } | null>) => void;
  disabled?: boolean;
}

const PassThroughGuardrailsSection: React.FC<PassThroughGuardrailsSectionProps> = ({
  accessToken,
  value = {},
  onChange,
  disabled = false,
}) => {
  const [selectedGuardrails, setSelectedGuardrails] = useState<string[]>(Object.keys(value));
  const [guardrailSettings, setGuardrailSettings] = useState<
    Record<string, { request_fields?: string[]; response_fields?: string[] } | null>
  >(value);

  // Sync external value changes
  useEffect(() => {
    setGuardrailSettings(value);
    setSelectedGuardrails(Object.keys(value));
  }, [value]);

  const handleGuardrailChange = (guardrails: string[]) => {
    setSelectedGuardrails(guardrails);

    // Create new settings object with selected guardrails
    const newSettings: Record<string, { request_fields?: string[]; response_fields?: string[] } | null> = {};
    guardrails.forEach((name) => {
      // Preserve existing settings or set to null (uses entire payload)
      newSettings[name] = guardrailSettings[name] || null;
    });

    setGuardrailSettings(newSettings);
    if (onChange) {
      onChange(newSettings);
    }
  };

  const handleFieldChange = (
    guardrailName: string,
    fieldType: "request_fields" | "response_fields",
    fields: string[]
  ) => {
    const currentSettings = guardrailSettings[guardrailName] || {};
    const newSettings = {
      ...guardrailSettings,
      [guardrailName]: {
        ...currentSettings,
        [fieldType]: fields.length > 0 ? fields : undefined,
      },
    };

    // If no fields are set, set to null (entire payload)
    if (!newSettings[guardrailName]?.request_fields && !newSettings[guardrailName]?.response_fields) {
      newSettings[guardrailName] = null;
    }

    setGuardrailSettings(newSettings);
    if (onChange) {
      onChange(newSettings);
    }
  };

  return (
    <Card className="p-6">
      <Title className="text-lg font-semibold text-gray-900 mb-2">Guardrails</Title>
      <Subtitle className="text-gray-600 mb-6">
        Configure guardrails to enforce policies on requests and responses. Guardrails are opt-in for passthrough
        endpoints.
      </Subtitle>

      <Alert
        message="Field-Level Targeting"
        description="You can specify which fields to send to each guardrail using JSONPath expressions (e.g., 'query', 'documents[*]'). If no fields are specified, the entire payload is sent."
        type="info"
        showIcon
        className="mb-4"
      />

      <Form.Item
        label={
          <span className="text-sm font-medium text-gray-700 flex items-center">
            Select Guardrails
            <Tooltip title="Choose which guardrails should run on this endpoint. Org/team/key level guardrails will also be included.">
              <InfoCircleOutlined className="ml-2 text-blue-400 hover:text-blue-600 cursor-help" />
            </Tooltip>
          </span>
        }
      >
        <GuardrailSelector
          accessToken={accessToken}
          value={selectedGuardrails}
          onChange={handleGuardrailChange}
          disabled={disabled}
        />
      </Form.Item>

      {selectedGuardrails.length > 0 && (
        <div className="mt-6 space-y-4">
          <div className="text-sm font-medium text-gray-700 mb-3">Field Targeting (Optional)</div>
          {selectedGuardrails.map((guardrailName) => (
            <Card key={guardrailName} className="p-4 bg-gray-50">
              <div className="text-sm font-medium text-gray-900 mb-3">{guardrailName}</div>
              <div className="space-y-3">
                <div>
                  <label className="text-xs text-gray-600 mb-1 block">
                    Request Fields
                    <Tooltip title="JSONPath expressions for fields to check in the request (e.g., 'query', 'documents[*].text')">
                      <InfoCircleOutlined className="ml-1 text-gray-400" />
                    </Tooltip>
                  </label>
                  <Select
                    mode="tags"
                    style={{ width: "100%" }}
                    placeholder="Leave empty to use entire request"
                    value={guardrailSettings[guardrailName]?.request_fields || []}
                    onChange={(fields) => handleFieldChange(guardrailName, "request_fields", fields)}
                    disabled={disabled}
                    tokenSeparators={[","]}
                  />
                </div>
                <div>
                  <label className="text-xs text-gray-600 mb-1 block">
                    Response Fields
                    <Tooltip title="JSONPath expressions for fields to check in the response (e.g., 'results[*].text')">
                      <InfoCircleOutlined className="ml-1 text-gray-400" />
                    </Tooltip>
                  </label>
                  <Select
                    mode="tags"
                    style={{ width: "100%" }}
                    placeholder="Leave empty to use entire response"
                    value={guardrailSettings[guardrailName]?.response_fields || []}
                    onChange={(fields) => handleFieldChange(guardrailName, "response_fields", fields)}
                    disabled={disabled}
                    tokenSeparators={[","]}
                  />
                </div>
              </div>
            </Card>
          ))}
        </div>
      )}
    </Card>
  );
};

export default PassThroughGuardrailsSection;

