import React, { useState } from "react";
import { Button } from "@tremor/react";
import { Input, Typography, Tooltip } from "antd";
import { CopyOutlined, InfoCircleOutlined } from "@ant-design/icons";
import { useTranslation, Trans } from "react-i18next";
import NotificationsManager from "../molecules/notifications_manager";
import GuardrailTestResults from "./GuardrailTestResults";

const { TextArea } = Input;
const { Text } = Typography;

interface GuardrailTestPanelProps {
  guardrailNames: string[];
  onSubmit: (text: string) => void;
  isLoading: boolean;
  results: Array<{ guardrailName: string; response_text: string; latency: number }> | null;
  errors: Array<{ guardrailName: string; error: Error; latency: number }> | null;
  onClose: () => void;
}

export function GuardrailTestPanel({
  guardrailNames,
  onSubmit,
  isLoading,
  results,
  errors,
  onClose,
}: GuardrailTestPanelProps) {
  const { t } = useTranslation();
  const [inputText, setInputText] = useState("");

  const handleSubmit = () => {
    if (!inputText.trim()) {
      NotificationsManager.fromBackend(t("guardrails.guardrailTestPanel.pleaseEnterText"));
      return;
    }

    onSubmit(inputText);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey && !e.ctrlKey && !e.metaKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const copyToClipboard = async (text: string) => {
    try {
      if (navigator.clipboard && window.isSecureContext) {
        await navigator.clipboard.writeText(text);
        return true;
      } else {
        const textArea = document.createElement("textarea");
        textArea.value = text;
        textArea.style.position = "fixed";
        textArea.style.opacity = "0";
        document.body.appendChild(textArea);
        textArea.focus();
        textArea.select();

        const successful = document.execCommand("copy");
        document.body.removeChild(textArea);

        if (!successful) {
          throw new Error("execCommand failed");
        }
        return true;
      }
    } catch (error) {
      console.error("Copy failed:", error);
      return false;
    }
  };

  const handleCopyInput = async () => {
    const success = await copyToClipboard(inputText);
    if (success) {
      NotificationsManager.success(t("guardrails.guardrailTestPanel.inputCopied"));
    } else {
      NotificationsManager.fromBackend(t("guardrails.guardrailTestPanel.failedToCopyInput"));
    }
  };

  return (
    <div className="space-y-4 h-full flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between pb-3 border-b border-gray-200">
        <div className="flex items-center space-x-3">
          <div className="flex-1 min-w-0">
            <div className="flex items-center space-x-2 mb-1">
              <h2 className="text-lg font-semibold text-gray-900">{t("guardrails.guardrailTestPanel.title")}</h2>
              <div className="flex flex-wrap gap-2">
                {guardrailNames.map((name) => (
                  <div
                    key={name}
                    className="inline-flex items-center space-x-1 bg-blue-50 px-3 py-1 rounded-md border border-blue-200"
                  >
                    <span className="font-mono text-blue-700 font-medium text-sm">{name}</span>
                  </div>
                ))}
              </div>
            </div>
            <p className="text-sm text-gray-500">
              {t("guardrails.guardrailTestPanel.subtitle", { count: guardrailNames.length })}
            </p>
          </div>
        </div>
      </div>

      {/* Input Section */}
      <div className="flex-1 overflow-auto space-y-4">
        <div className="space-y-3">
          <div>
            <div className="flex justify-between items-center mb-2">
              <div className="flex items-center gap-2">
                <label className="text-sm font-medium text-gray-700">
                  {t("guardrails.guardrailTestPanel.inputText")}
                </label>
                <Tooltip title={t("guardrails.guardrailTestPanel.inputTooltip")}>
                  <InfoCircleOutlined className="text-gray-400 cursor-help" />
                </Tooltip>
              </div>
              {inputText && (
                <Button size="xs" variant="secondary" icon={CopyOutlined} onClick={handleCopyInput}>
                  {t("guardrails.guardrailTestPanel.copyInput")}
                </Button>
              )}
            </div>
            <TextArea
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={t("guardrails.guardrailTestPanel.inputPlaceholder")}
              rows={8}
              className="font-mono text-sm"
            />
            <div className="flex justify-between items-center mt-1">
              <Text className="text-xs text-gray-500">
                <Trans
                  i18nKey="guardrails.guardrailTestPanel.keyboardHint"
                  components={{
                    enter: <kbd className="px-1 py-0.5 bg-gray-100 border border-gray-300 rounded text-xs" />,
                    shiftEnter: <kbd className="px-1 py-0.5 bg-gray-100 border border-gray-300 rounded text-xs" />,
                  }}
                />
              </Text>
              <Text className="text-xs text-gray-500">
                {t("guardrails.guardrailTestPanel.characters", { count: inputText.length })}
              </Text>
            </div>
          </div>

          <div className="pt-2">
            <Button onClick={handleSubmit} loading={isLoading} disabled={!inputText.trim()} className="w-full">
              {isLoading
                ? t("guardrails.guardrailTestPanel.testingButton", { count: guardrailNames.length })
                : t("guardrails.guardrailTestPanel.testButton", { count: guardrailNames.length })}
            </Button>
          </div>
        </div>

        {/* Results Section */}
        <GuardrailTestResults results={results} errors={errors} />
      </div>
    </div>
  );
}

export default GuardrailTestPanel;
