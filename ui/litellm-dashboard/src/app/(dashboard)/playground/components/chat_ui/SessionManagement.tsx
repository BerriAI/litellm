import React from "react";
import { Copy, Info } from "lucide-react";
import { useTranslation } from "react-i18next";
import { EndpointType } from "@/components/chat_ui/mode_endpoint_mapping";
import { toast } from "@/lib/toast";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { codeSamples } from "@/i18n/untranslated/codeSamples";

interface SessionManagementProps {
  endpointType: string | null;
  responsesSessionId: string | null;
  useApiSessionManagement: boolean;
  onToggleSessionManagement: (useApi: boolean) => void;
}

const SessionManagement: React.FC<SessionManagementProps> = ({
  endpointType,
  responsesSessionId,
  useApiSessionManagement,
  onToggleSessionManagement,
}) => {
  const { t } = useTranslation();
  if (endpointType !== EndpointType.RESPONSES) {
    return null;
  }

  const handleCopySessionId = async () => {
    if (responsesSessionId) {
      try {
        await navigator.clipboard.writeText(responsesSessionId);
        toast.success(t("playground.chatUi.sessionManagement.copied"));
      } catch {
        toast.error(t("playground.chatUi.sessionManagement.copyFailed"));
      }
    }
  };

  const getSessionDisplay = () => {
    if (!responsesSessionId) {
      return useApiSessionManagement
        ? t("playground.chatUi.sessionManagement.apiReady")
        : t("playground.chatUi.sessionManagement.uiReady");
    }

    const sessionPrefix = useApiSessionManagement
      ? t("playground.chatUi.sessionManagement.responseIdPrefix")
      : t("playground.chatUi.sessionManagement.uiSessionPrefix");
    const truncatedId = responsesSessionId.slice(0, 10);
    return `${sessionPrefix}: ${truncatedId}...`;
  };

  const getSessionDescription = () => {
    if (!responsesSessionId) {
      return useApiSessionManagement
        ? t("playground.chatUi.sessionManagement.apiPending")
        : t("playground.chatUi.sessionManagement.uiPending");
    }

    return useApiSessionManagement
      ? t("playground.chatUi.sessionManagement.apiActive")
      : t("playground.chatUi.sessionManagement.uiActive");
  };

  return (
    <div className="mb-4">
      {/* Session Management Toggle */}
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium text-foreground">{t("playground.chatUi.sessionManagement.title")}</span>
          <Tooltip>
            <TooltipTrigger aria-label={t("playground.chatUi.sessionManagement.about")}>
              <Info className="size-3 text-muted-foreground" />
            </TooltipTrigger>
            <TooltipContent>{t("playground.chatUi.sessionManagement.description")}</TooltipContent>
          </Tooltip>
        </div>
        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          <span aria-hidden="true">{t("playground.chatUi.sessionManagement.uiLabel")}</span>
          <Switch
            checked={useApiSessionManagement}
            onCheckedChange={onToggleSessionManagement}
            aria-label={t("playground.chatUi.sessionManagement.toggle")}
            size="sm"
          />
          <span aria-hidden="true">{t("playground.chatUi.sessionManagement.apiLabel")}</span>
        </div>
      </div>

      {/* Session Status Indicator */}
      <div
        className={`text-xs p-2 rounded-md ${
          responsesSessionId
            ? "bg-success/10 text-success border border-success/20"
            : "bg-info/10 text-info border border-info/20"
        }`}
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-1">
            <Info className="size-3" />
            {getSessionDisplay()}
          </div>
          {responsesSessionId && (
            <Tooltip>
              <TooltipTrigger
                render={
                  <Button
                    type="button"
                    variant="ghost"
                    size="icon-xs"
                    onClick={handleCopySessionId}
                    aria-label={t("playground.chatUi.sessionManagement.copyResponseId")}
                    className="ml-2 hover:bg-success/15"
                  />
                }
              >
                <Copy className="size-3" />
              </TooltipTrigger>
              <TooltipContent className="max-w-lg">
                <div className="text-xs">
                  <div className="mb-1">{t("playground.chatUi.sessionManagement.copyToContinue")}</div>
                  <div className="bg-gray-800 text-gray-100 p-2 rounded-sm font-mono text-xs whitespace-pre-wrap">
                    {codeSamples.playground.responsesSessionCurl(responsesSessionId)}
                  </div>
                </div>
              </TooltipContent>
            </Tooltip>
          )}
        </div>
        <div className="text-xs opacity-75 mt-1">{getSessionDescription()}</div>
      </div>
    </div>
  );
};

export default SessionManagement;
