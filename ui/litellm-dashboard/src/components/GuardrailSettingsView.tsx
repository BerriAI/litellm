import React from "react";
import { Badge, Text } from "@tremor/react";
import { GlobalOutlined } from "@ant-design/icons";

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
  const optedOutSet = new Set(optedOutGlobalGuardrails);
  const globalsRunning = Array.from(globalGuardrailNames).filter(
    (n) => !optedOutSet.has(n),
  );
  const nonGlobalOptIns = teamGuardrails.filter(
    (n) => !globalGuardrailNames.has(n),
  );

  const isEmpty =
    !killSwitchOn && globalsRunning.length === 0 && nonGlobalOptIns.length === 0;

  const content = isEmpty ? (
    <Text className="text-gray-500">No guardrails configured</Text>
  ) : (
    <div className="flex flex-col gap-4">
      <div>
        <Text className="text-sm font-medium text-gray-700 mb-2">
          <GlobalOutlined style={{ marginInlineEnd: 4 }} aria-label="Global guardrail" />
          Global
        </Text>
        {killSwitchOn ? (
          <Badge color="yellow">Bypassed for this team</Badge>
        ) : globalsRunning.length > 0 ? (
          <div className="flex flex-wrap gap-2">
            {globalsRunning.map((name) => (
              <Badge key={name} color="blue">
                {name}
              </Badge>
            ))}
          </div>
        ) : (
          <Text className="text-gray-500 text-sm">None configured</Text>
        )}
      </div>
      <div>
        <Text className="text-sm font-medium text-gray-700 mb-2">Team-specific</Text>
        {nonGlobalOptIns.length > 0 ? (
          <div className="flex flex-wrap gap-2">
            {nonGlobalOptIns.map((name) => (
              <Badge key={name} color="blue">
                {name}
              </Badge>
            ))}
          </div>
        ) : (
          <Text className="text-gray-500 text-sm">None configured</Text>
        )}
      </div>
    </div>
  );

  if (variant === "card") {
    return (
      <div className={`bg-white border border-gray-200 rounded-lg p-6 ${className}`}>
        <div className="flex items-center gap-2 mb-6">
          <div>
            <span className="block font-semibold text-gray-900">Guardrails Settings</span>
            <span className="block text-xs text-gray-500">
              Global and team-specific guardrails applied to this team
            </span>
          </div>
        </div>
        {content}
      </div>
    );
  }

  return (
    <div className={`${className}`}>
      <span className="block font-medium text-gray-900 mb-3">Guardrails Settings</span>
      {content}
    </div>
  );
}

export default GuardrailSettingsView;
