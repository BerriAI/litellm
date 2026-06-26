import React, { useState, useEffect } from "react";
import { Card, Title, Subtitle } from "@tremor/react";
import { Form, Select, Tooltip, Alert } from "antd";
import { InfoCircleOutlined } from "@ant-design/icons";
import { useTranslation } from "react-i18next";
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
  const { t } = useTranslation();
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
      <Title className="text-lg font-semibold text-gray-900 mb-2">
        {t("commonComponents.passThroughGuardrailsSection.title")}
      </Title>
      <Subtitle className="text-gray-600 mb-6">{t("commonComponents.passThroughGuardrailsSection.subtitle")}</Subtitle>

      <Alert
        message={
          <span>
            {t("commonComponents.passThroughGuardrailsSection.fieldLevelTargeting")}{" "}
            <a
              href="https://docs.litellm.ai/docs/proxy/pass_through_guardrails#field-level-targeting"
              target="_blank"
              rel="noopener noreferrer"
              className="text-blue-600 hover:text-blue-800 underline"
            >
              ({t("common.learnMore")})
            </a>
          </span>
        }
        description={
          <div className="space-y-2">
            <div>{t("commonComponents.passThroughGuardrailsSection.fieldLevelDesc")}</div>
            <div className="text-xs space-y-1 mt-2">
              <div className="font-medium">{t("commonComponents.passThroughGuardrailsSection.commonExamples")}</div>
              <div>
                • <code className="bg-gray-100 px-1 rounded">query</code> -{" "}
                {t("commonComponents.passThroughGuardrailsSection.singleField")}
              </div>
              <div>
                • <code className="bg-gray-100 px-1 rounded">documents[*].text</code> -{" "}
                {t("commonComponents.passThroughGuardrailsSection.allTextInDocuments")}
              </div>
              <div>
                • <code className="bg-gray-100 px-1 rounded">messages[*].content</code> -{" "}
                {t("commonComponents.passThroughGuardrailsSection.allMessageContents")}
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
            {t("commonComponents.passThroughGuardrailsSection.selectGuardrails")}
            <Tooltip title={t("commonComponents.passThroughGuardrailsSection.selectGuardrailsTooltip")}>
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
            <div className="text-sm font-medium text-gray-700">
              {t("commonComponents.passThroughGuardrailsSection.fieldTargetingOptional")}
            </div>
            <div className="text-xs text-gray-500">
              {t("commonComponents.passThroughGuardrailsSection.leaveEmptyTip")}
            </div>
          </div>
          {selectedGuardrails.map((guardrailName) => (
            <Card key={guardrailName} className="p-4 bg-gray-50">
              <div className="text-sm font-medium text-gray-900 mb-3">{guardrailName}</div>
              <div className="space-y-3">
                <div>
                  <div className="flex items-center justify-between mb-1">
                    <label className="text-xs text-gray-600 flex items-center">
                      {t("commonComponents.passThroughGuardrailsSection.requestFields")}
                      <Tooltip
                        title={
                          <div>
                            <div className="font-medium mb-1">
                              {t("commonComponents.passThroughGuardrailsSection.requestFieldsTooltipTitle")}
                            </div>
                            <div className="text-xs space-y-1">
                              <div>{t("commonComponents.passThroughGuardrailsSection.examples")}</div>
                              <div>{t("commonComponents.passThroughGuardrailsSection.tooltipExampleQuery")}</div>
                              <div>{t("commonComponents.passThroughGuardrailsSection.tooltipExampleDocuments")}</div>
                              <div>{t("commonComponents.passThroughGuardrailsSection.tooltipExampleMessages")}</div>
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
                        className="text-xs px-2 py-1 bg-white border border-gray-300 rounded hover:bg-gray-50"
                        disabled={disabled}
                      >
                        {t("commonComponents.passThroughGuardrailsSection.addQueryButton")}
                      </button>
                      <button
                        type="button"
                        onClick={() => {
                          const current = guardrailSettings[guardrailName]?.request_fields || [];
                          handleFieldChange(guardrailName, "request_fields", [...current, "documents[*]"]);
                        }}
                        className="text-xs px-2 py-1 bg-white border border-gray-300 rounded hover:bg-gray-50"
                        disabled={disabled}
                      >
                        {t("commonComponents.passThroughGuardrailsSection.addDocumentsButton")}
                      </button>
                    </div>
                  </div>
                  <Select
                    mode="tags"
                    style={{ width: "100%" }}
                    placeholder={t("commonComponents.passThroughGuardrailsSection.requestFieldsPlaceholder")}
                    value={guardrailSettings[guardrailName]?.request_fields || []}
                    onChange={(fields) => handleFieldChange(guardrailName, "request_fields", fields)}
                    disabled={disabled}
                    tokenSeparators={[","]}
                  />
                </div>
                <div>
                  <div className="flex items-center justify-between mb-1">
                    <label className="text-xs text-gray-600 flex items-center">
                      {t("commonComponents.passThroughGuardrailsSection.responseFields")}
                      <Tooltip
                        title={
                          <div>
                            <div className="font-medium mb-1">
                              {t("commonComponents.passThroughGuardrailsSection.responseFieldsTooltipTitle")}
                            </div>
                            <div className="text-xs space-y-1">
                              <div>{t("commonComponents.passThroughGuardrailsSection.examples")}</div>
                              <div>{t("commonComponents.passThroughGuardrailsSection.tooltipExampleResults")}</div>
                              <div>{t("commonComponents.passThroughGuardrailsSection.tooltipExampleChoices")}</div>
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
                        className="text-xs px-2 py-1 bg-white border border-gray-300 rounded hover:bg-gray-50"
                        disabled={disabled}
                      >
                        {t("commonComponents.passThroughGuardrailsSection.addResultsButton")}
                      </button>
                    </div>
                  </div>
                  <Select
                    mode="tags"
                    style={{ width: "100%" }}
                    placeholder={t("commonComponents.passThroughGuardrailsSection.responseFieldsPlaceholder")}
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
