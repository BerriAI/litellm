"use client";

import { FileText, ScrollText } from "lucide-react";
import type { ParseKeys } from "i18next";
import { useTranslation } from "react-i18next";

import { callback_map } from "@/components/callback_info_helpers";

import { SectionCard, SubHead } from "./SectionCard";

export interface LoggingIntegration {
  readonly callback_name: string;
  readonly callback_type: string;
  readonly callback_vars: Readonly<Record<string, string>>;
}

interface LoggingCardProps {
  integrations: readonly LoggingIntegration[];
  disabledCallbacks: readonly string[];
}

const EVENT_TYPE_KEYS: Readonly<Record<string, ParseKeys>> = {
  success: "loggingSettingsView.eventTypes.success",
  failure: "loggingSettingsView.eventTypes.failure",
  success_and_failure: "loggingSettingsView.eventTypes.successAndFailure",
};

const displayName = (callbackName: string): string =>
  Object.entries(callback_map).find(([, value]) => value === callbackName)?.[0] ?? callbackName;

export function LoggingCard({ integrations, disabledCallbacks }: LoggingCardProps) {
  const { t } = useTranslation();
  return (
    <SectionCard icon={ScrollText} title={t("keyDetail.overview.logging.title")} testId="key-logging-card">
      {integrations.length > 0 ? (
        <div className="flex flex-col gap-2">
          <SubHead title={t("keyDetail.overview.logging.integrations")} count={integrations.length} />
          {integrations.map((integration, index) => {
            const eventKey = EVENT_TYPE_KEYS[integration.callback_type];
            return (
              <div
                key={`${integration.callback_name}-${index}`}
                className="flex flex-wrap items-center gap-x-2 gap-y-1.5"
              >
                <FileText className="size-4 text-muted-foreground" aria-hidden />
                <span className="font-medium">{displayName(integration.callback_name)}</span>
                <span className="text-muted-foreground">
                  {t("keyDetail.overview.logging.integrationDetail", {
                    events: eventKey ? t(eventKey) : integration.callback_type,
                    count: Object.keys(integration.callback_vars ?? {}).length,
                  })}
                </span>
              </div>
            );
          })}
        </div>
      ) : null}
      <div className="border-t border-border pt-3 text-muted-foreground">
        {disabledCallbacks.length > 0 ? (
          <>
            {t("keyDetail.overview.logging.turnedOff")}
            <span className="font-medium text-foreground">{disabledCallbacks.join(", ")}</span>
          </>
        ) : (
          t("keyDetail.overview.logging.noneTurnedOff")
        )}
      </div>
    </SectionCard>
  );
}
