import React from "react";
import { Tag } from "antd";
import { GlobalOutlined } from "@ant-design/icons";
import { useTranslation } from "react-i18next";

interface GuardrailSettingsViewProps {
  globalGuardrailNames: Set<string>;
  teamGuardrails?: string[];
  optedOutGlobalGuardrails?: string[];
  killSwitchOn?: boolean;
  variant?: "card" | "inline";
  className?: string;
}

export function GuardrailSettingsView({
  globalGuardrailNames,
  teamGuardrails = [],
  optedOutGlobalGuardrails = [],
  killSwitchOn = false,
  variant = "card",
  className = "",
}: GuardrailSettingsViewProps) {
  const { t } = useTranslation();
  const optedOutSet = new Set(optedOutGlobalGuardrails);
  const globalsRunning = Array.from(globalGuardrailNames).filter((n) => !optedOutSet.has(n));
  const nonGlobalOptIns = teamGuardrails.filter((n) => !globalGuardrailNames.has(n));

  const isEmpty = !killSwitchOn && globalsRunning.length === 0 && nonGlobalOptIns.length === 0;

  const content = isEmpty ? (
    <span className="block text-gray-500">{t("guardrailSettingsView.noGuardrailsConfigured")}</span>
  ) : (
    <div className="flex flex-col gap-4">
      <div>
        <span className="block text-sm font-medium text-gray-700 mb-2">
          <GlobalOutlined
            style={{ marginInlineEnd: 4 }}
            aria-label={t("guardrailSettingsView.globalGuardrailAriaLabel")}
          />
          {t("guardrailSettingsView.globalLabel")}
        </span>
        {killSwitchOn ? (
          <Tag color="gold">{t("guardrailSettingsView.bypassedForTeam")}</Tag>
        ) : globalsRunning.length > 0 ? (
          <div className="flex flex-wrap gap-2">
            {globalsRunning.map((name) => (
              <Tag key={name} color="blue">
                {name}
              </Tag>
            ))}
          </div>
        ) : (
          <span className="block text-sm text-gray-500">{t("guardrailSettingsView.noneConfigured")}</span>
        )}
      </div>
      <div>
        <span className="block text-sm font-medium text-gray-700 mb-2">
          {t("guardrailSettingsView.teamSpecificLabel")}
        </span>
        {nonGlobalOptIns.length > 0 ? (
          <div className="flex flex-wrap gap-2">
            {nonGlobalOptIns.map((name) => (
              <Tag key={name} color="blue">
                {name}
              </Tag>
            ))}
          </div>
        ) : (
          <span className="block text-sm text-gray-500">{t("guardrailSettingsView.noneConfigured")}</span>
        )}
      </div>
    </div>
  );

  if (variant === "card") {
    return (
      <div className={`bg-white border border-gray-200 rounded-lg p-6 ${className}`}>
        <div className="flex items-center gap-2 mb-6">
          <div>
            <span className="block font-semibold text-gray-900">{t("guardrailSettingsView.cardTitle")}</span>
            <span className="block text-xs text-gray-500">{t("guardrailSettingsView.cardSubtitle")}</span>
          </div>
        </div>
        {content}
      </div>
    );
  }

  return (
    <div className={`${className}`}>
      <span className="block font-medium text-gray-900 mb-3">{t("guardrailSettingsView.inlineTitle")}</span>
      {content}
    </div>
  );
}

export default GuardrailSettingsView;
