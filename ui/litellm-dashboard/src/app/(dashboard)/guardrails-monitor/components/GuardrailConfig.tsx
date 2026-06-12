import {
  CheckCircleOutlined,
  CodeOutlined,
  PlayCircleOutlined,
  RollbackOutlined,
  SaveOutlined,
} from "@ant-design/icons";
import { Button, Input, Select, Switch } from "antd";
import React, { useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import type { TFunction } from "i18next";

interface GuardrailConfigProps {
  guardrailName: string;
  guardrailType: string;
  provider: string;
}

const versionsBase = [
  {
    id: "v3",
    date: "2026-02-18",
    author: "admin@company.com",
    changes: "Adjusted sensitivity for medical terms",
  },
  { id: "v2", date: "2026-02-10", author: "admin@company.com", changes: "Added custom categories list" },
  { id: "v1", date: "2026-01-28", author: "admin@company.com", changes: "Initial configuration" },
];

const getVersions = (t: TFunction, currentId: string) =>
  versionsBase.map((v) => ({
    ...v,
    label: v.id === currentId ? `${v.id} (${t("guardrailsMonitor.guardrailConfig.current")})` : v.id,
  }));

const getActionOptions = (t: TFunction) => [
  { value: "block", label: t("guardrailsMonitor.guardrailConfig.actionBlock") },
  { value: "flag", label: t("guardrailsMonitor.guardrailConfig.actionFlag") },
  { value: "log", label: t("guardrailsMonitor.guardrailConfig.actionLog") },
  { value: "fallback", label: t("guardrailsMonitor.guardrailConfig.actionFallback") },
];

const getProviderOptions = (t: TFunction) => [
  { value: "bedrock", label: t("guardrailsMonitor.guardrailConfig.providerBedrock") },
  { value: "google", label: t("guardrailsMonitor.guardrailConfig.providerGoogle") },
  { value: "litellm", label: t("guardrailsMonitor.guardrailConfig.providerLiteLLM") },
  { value: "custom", label: t("guardrailsMonitor.guardrailConfig.providerCustom") },
];

const getGuardrailTypeOptions = (t: TFunction) => [
  { value: "Content Safety", label: t("guardrailsMonitor.guardrailConfig.typeContentSafety") },
  { value: "PII", label: t("guardrailsMonitor.guardrailConfig.typePII") },
  { value: "Topic", label: t("guardrailsMonitor.guardrailConfig.typeTopic") },
  { value: "prompt_injection", label: t("guardrailsMonitor.guardrailConfig.typePromptInjection") },
  { value: "custom", label: t("guardrailsMonitor.guardrailConfig.typeCustom") },
];

export function GuardrailConfig({ guardrailName, guardrailType, provider }: GuardrailConfigProps) {
  const { t } = useTranslation();
  const [action, setAction] = useState("block");
  const [enabled, setEnabled] = useState(true);
  const [customCode, setCustomCode] = useState("");
  const [useCustomCode, setUseCustomCode] = useState(false);
  const [rerunStatus, setRerunStatus] = useState<"idle" | "running" | "success" | "error">("idle");
  const [version, setVersion] = useState("v3");
  const [showVersionHistory, setShowVersionHistory] = useState(false);

  const versions = useMemo(() => getVersions(t, version), [t, version]);
  const actionOptions = useMemo(() => getActionOptions(t), [t]);
  const providerOptions = useMemo(() => getProviderOptions(t), [t]);
  const guardrailTypeOptions = useMemo(() => getGuardrailTypeOptions(t), [t]);

  const handleRerun = () => {
    setRerunStatus("running");
    setTimeout(() => {
      setRerunStatus("success");
      setTimeout(() => setRerunStatus("idle"), 3000);
    }, 2000);
  };

  return (
    <div className="space-y-6">
      {/* Version Bar */}
      <div className="bg-white border border-gray-200 rounded-lg p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <span className="text-sm font-medium text-gray-700">
              {t("guardrailsMonitor.guardrailConfig.versionLabel")}
            </span>
            <Select
              value={version}
              onChange={setVersion}
              options={versions.map((v) => ({ value: v.id, label: v.label }))}
              style={{ width: 140 }}
            />
            <Button type="link" size="small" onClick={() => setShowVersionHistory(!showVersionHistory)}>
              {showVersionHistory
                ? t("guardrailsMonitor.guardrailConfig.hideHistory")
                : t("guardrailsMonitor.guardrailConfig.viewHistory")}
            </Button>
          </div>
          <div className="flex items-center gap-2">
            <Button icon={<RollbackOutlined />}>{t("guardrailsMonitor.guardrailConfig.revert")}</Button>
            <Button type="primary" icon={<SaveOutlined />}>
              {t("guardrailsMonitor.guardrailConfig.saveAsVersion", {
                version: parseInt(version.replace("v", ""), 10) + 1,
              })}
            </Button>
          </div>
        </div>

        {showVersionHistory && (
          <div className="mt-4 border-t border-gray-100 pt-4 space-y-2">
            {versions.map((v) => (
              <div
                key={v.id}
                className={`flex items-center justify-between p-2.5 rounded-md text-sm ${
                  v.id === version ? "bg-blue-50 border border-blue-200" : "bg-gray-50"
                }`}
              >
                <div className="flex items-center gap-3">
                  <span
                    className={`font-mono text-xs font-medium ${v.id === version ? "text-blue-600" : "text-gray-500"}`}
                  >
                    {v.id}
                  </span>
                  <span className="text-gray-700">{v.changes}</span>
                </div>
                <div className="flex items-center gap-3 text-xs text-gray-500">
                  <span>{v.author}</span>
                  <span>{v.date}</span>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Parameters */}
      <div className="bg-white border border-gray-200 rounded-lg p-6">
        <h3 className="text-base font-semibold text-gray-900 mb-1">
          {t("guardrailsMonitor.guardrailConfig.parametersTitle")}
        </h3>
        <p className="text-xs text-gray-500 mb-5">
          {t("guardrailsMonitor.guardrailConfig.parametersDesc", { name: guardrailName })}
        </p>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1.5">
              {t("guardrailsMonitor.guardrailConfig.actionOnFailure")}
            </label>
            <Select value={action} onChange={setAction} style={{ width: "100%" }} options={actionOptions} />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1.5">
              {t("guardrailsMonitor.guardrailConfig.providerLabel")}
            </label>
            <Select style={{ width: "100%" }} defaultValue={provider} options={providerOptions} />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1.5">
              {t("guardrailsMonitor.guardrailConfig.guardrailTypeLabel")}
            </label>
            <Select style={{ width: "100%" }} defaultValue={guardrailType} options={guardrailTypeOptions} />
          </div>

          <div className="md:col-span-2">
            <label className="block text-sm font-medium text-gray-700 mb-1.5">
              {t("guardrailsMonitor.guardrailConfig.categoriesLabel")}
            </label>
            <Input defaultValue="violence, hate_speech, sexual_content, self_harm, illegal_activity" />
          </div>

          <div className="md:col-span-2 flex items-center gap-3">
            <Switch checked={enabled} onChange={setEnabled} />
            <span className="text-sm text-gray-700">{t("guardrailsMonitor.guardrailConfig.enabledInProduction")}</span>
          </div>
        </div>
      </div>

      {/* Custom Code Override */}
      <div className="bg-white border border-gray-200 rounded-lg p-6">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="text-base font-semibold text-gray-900 flex items-center gap-2">
              <CodeOutlined className="text-gray-500" />
              {t("guardrailsMonitor.guardrailConfig.customCodeTitle")}
            </h3>
            <p className="text-xs text-gray-500 mt-0.5">{t("guardrailsMonitor.guardrailConfig.customCodeDesc")}</p>
          </div>
          <Switch checked={useCustomCode} onChange={setUseCustomCode} />
        </div>

        {useCustomCode && (
          <Input.TextArea
            value={customCode}
            onChange={(e) => setCustomCode(e.target.value)}
            placeholder={`async def evaluate(input_text: str, context: dict) -> dict:
    # Return {"score": 0.0-1.0, "passed": bool, "reason": str}
    # Example:
    if "banned_word" in input_text.lower():
        return {"score": 0.1, "passed": False, "reason": "Banned word detected"}
    return {"score": 0.9, "passed": True, "reason": "No violations"}`}
            rows={10}
            className="font-mono text-sm"
          />
        )}
      </div>

      {/* Re-run on Failing Logs */}
      <div className="bg-white border border-gray-200 rounded-lg p-6">
        <h3 className="text-base font-semibold text-gray-900 mb-1">
          {t("guardrailsMonitor.guardrailConfig.testConfigTitle")}
        </h3>
        <p className="text-xs text-gray-500 mb-4">{t("guardrailsMonitor.guardrailConfig.testConfigDesc")}</p>

        <div className="flex items-center gap-3">
          <Button
            type="primary"
            icon={rerunStatus === "running" ? undefined : <PlayCircleOutlined />}
            loading={rerunStatus === "running"}
            onClick={handleRerun}
          >
            {rerunStatus === "running"
              ? t("guardrailsMonitor.guardrailConfig.runningOnSamples")
              : t("guardrailsMonitor.guardrailConfig.rerunOnFailingLogs")}
          </Button>

          {rerunStatus === "success" && (
            <span className="text-sm text-green-600 flex items-center gap-2">
              <CheckCircleOutlined /> {t("guardrailsMonitor.guardrailConfig.rerunSuccess")}
            </span>
          )}

          {rerunStatus === "error" && (
            <span className="text-sm text-red-600">{t("guardrailsMonitor.guardrailConfig.rerunError")}</span>
          )}
        </div>
      </div>
    </div>
  );
}
