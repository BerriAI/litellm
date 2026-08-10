import React, { useState, useEffect } from "react";
import { Card, Title, Subtitle } from "@tremor/react";
import { Form, Select, Tooltip, Alert } from "antd";
import { InfoCircleOutlined } from "@ant-design/icons";
import GuardrailSelector from "../guardrails/GuardrailSelector";
import { useTranslation } from "react-i18next";

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
  const { t } = useTranslation("gateway");
  const [selectedGuardrails, setSelectedGuardrails] = useState<string[]>(Object.keys(value));
  const [guardrailSettings, setGuardrailSettings] =
    useState<Record<string, { request_fields?: string[]; response_fields?: string[] } | null>>(value);

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
    fields: string[],
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
      <Title className="text-lg font-semibold text-gray-900 mb-2">{t("models.passThrough.guardrails.title")}</Title>
      <Subtitle className="text-gray-600 mb-6">{t("models.passThrough.guardrails.description")}</Subtitle>

      <Alert
        message={
          <span>
            {t("models.passThrough.guardrails.targeting")}{" "}
            <a
              href="https://docs.litellm.ai/docs/proxy/pass_through_guardrails#field-level-targeting"
              target="_blank"
              rel="noopener noreferrer"
              className="text-blue-600 hover:text-blue-800 underline"
            >
              ({t("models.passThrough.guardrails.learnMore")})
            </a>
          </span>
        }
        description={
          <div className="space-y-2">
            <div>{t("models.passThrough.guardrails.targetingDescription")}</div>
            <div className="text-xs space-y-1 mt-2">
              <div className="font-medium">{t("models.passThrough.guardrails.examples")}</div>
              <div>
                • <code className="bg-gray-100 px-1 rounded-sm">query</code> —{" "}
                {t("models.passThrough.guardrails.singleField")}
              </div>
              <div>
                • <code className="bg-gray-100 px-1 rounded-sm">documents[*].text</code> —{" "}
                {t("models.passThrough.guardrails.documentsArray")}
              </div>
              <div>
                • <code className="bg-gray-100 px-1 rounded-sm">messages[*].content</code> —{" "}
                {t("models.passThrough.guardrails.messageContents")}
              </div>
            </div>
          </div>
        }
        type="info"
        showIcon
        className="mb-4"
      />

      <Form.Item
        label={
          <span className="text-sm font-medium text-gray-700 flex items-center">
            {t("models.passThrough.guardrails.select")}
            <Tooltip title={t("models.passThrough.guardrails.selectTooltip")}>
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
          <div className="flex items-center justify-between mb-3">
            <div className="text-sm font-medium text-gray-700">{t("models.passThrough.guardrails.fieldTargeting")}</div>
            <div className="text-xs text-gray-500">{t("models.passThrough.guardrails.tip")}</div>
          </div>
          {selectedGuardrails.map((guardrailName) => (
            <Card key={guardrailName} className="p-4 bg-gray-50">
              <div className="text-sm font-medium text-gray-900 mb-3">{guardrailName}</div>
              <div className="space-y-3">
                <div>
                  <div className="flex items-center justify-between mb-1">
                    <label className="text-xs text-gray-600 flex items-center">
                      {t("models.passThrough.guardrails.requestFields")}
                      <Tooltip
                        title={
                          <div>
                            <div className="font-medium mb-1">{t("models.passThrough.guardrails.requestTooltip")}</div>
                            <div className="text-xs space-y-1">
                              <div>{t("models.passThrough.guardrails.examples")}</div>
                              <div>• query</div>
                              <div>• documents[*].text</div>
                              <div>• messages[*].content</div>
                            </div>
                          </div>
                        }
                      >
                        <InfoCircleOutlined className="ml-1 text-gray-400" />
                      </Tooltip>
                    </label>
                    <div className="flex gap-1">
                      <button
                        type="button"
                        onClick={() => {
                          const current = guardrailSettings[guardrailName]?.request_fields || [];
                          handleFieldChange(guardrailName, "request_fields", [...current, "query"]);
                        }}
                        className="text-xs px-2 py-1 bg-white border border-gray-300 rounded-sm hover:bg-gray-50"
                        disabled={disabled}
                      >
                        + query
                      </button>
                      <button
                        type="button"
                        onClick={() => {
                          const current = guardrailSettings[guardrailName]?.request_fields || [];
                          handleFieldChange(guardrailName, "request_fields", [...current, "documents[*]"]);
                        }}
                        className="text-xs px-2 py-1 bg-white border border-gray-300 rounded-sm hover:bg-gray-50"
                        disabled={disabled}
                      >
                        + documents[*]
                      </button>
                    </div>
                  </div>
                  <Select
                    mode="tags"
                    style={{ width: "100%" }}
                    placeholder={t("models.passThrough.guardrails.requestPlaceholder")}
                    value={guardrailSettings[guardrailName]?.request_fields || []}
                    onChange={(fields) => handleFieldChange(guardrailName, "request_fields", fields)}
                    disabled={disabled}
                    tokenSeparators={[","]}
                  />
                </div>
                <div>
                  <div className="flex items-center justify-between mb-1">
                    <label className="text-xs text-gray-600 flex items-center">
                      {t("models.passThrough.guardrails.responseFields")}
                      <Tooltip
                        title={
                          <div>
                            <div className="font-medium mb-1">{t("models.passThrough.guardrails.responseTooltip")}</div>
                            <div className="text-xs space-y-1">
                              <div>{t("models.passThrough.guardrails.examples")}</div>
                              <div>• results[*].text</div>
                              <div>• choices[*].message.content</div>
                            </div>
                          </div>
                        }
                      >
                        <InfoCircleOutlined className="ml-1 text-gray-400" />
                      </Tooltip>
                    </label>
                    <div className="flex gap-1">
                      <button
                        type="button"
                        onClick={() => {
                          const current = guardrailSettings[guardrailName]?.response_fields || [];
                          handleFieldChange(guardrailName, "response_fields", [...current, "results[*]"]);
                        }}
                        className="text-xs px-2 py-1 bg-white border border-gray-300 rounded-sm hover:bg-gray-50"
                        disabled={disabled}
                      >
                        + results[*]
                      </button>
                    </div>
                  </div>
                  <Select
                    mode="tags"
                    style={{ width: "100%" }}
                    placeholder={t("models.passThrough.guardrails.responsePlaceholder")}
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
