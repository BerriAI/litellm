import type { TFunction } from "i18next";
import React, { useEffect, useState } from "react";
import { Trans, useTranslation } from "react-i18next";
import { Ban, Clock3, Cloud, Database, Info, LoaderCircle, RefreshCw, TriangleAlert } from "lucide-react";

import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { InputGroup, InputGroupAddon, InputGroupInput } from "@/components/ui/input-group";
import { Separator } from "@/components/ui/separator";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import i18n from "@/i18n/i18n";
import { cn } from "@/lib/cva.config";

import { toast } from "@/lib/toast";
import {
  cancelModelCostMapReload,
  getModelCostMapReloadStatus,
  getModelCostMapSource,
  reloadModelCostMap,
  scheduleModelCostMapReload,
} from "./networking";

interface ReloadStatus {
  scheduled: boolean;
  interval_hours: number | null;
  last_run: string | null;
  next_run: string | null;
}

interface CostMapSourceInfo {
  source: "local" | "remote";
  url: string | null;
  is_env_forced: boolean;
  fallback_reason: string | null;
  loaded_at: string | null;
  source_revision: string | null;
  etag: string | null;
  model_count: number;
}

const SHORT_REVISION_LENGTH = 12;

const shortRevision = (revision: string) => revision.slice(0, SHORT_REVISION_LENGTH);

const EMPTY_RELOAD_STATUS: ReloadStatus = {
  scheduled: false,
  interval_hours: null,
  last_run: null,
  next_run: null,
};

interface PriceDataReloadProps {
  accessToken: string;
  onReloadSuccess?: () => void;
  buttonText?: string;
  showIcon?: boolean;
  size?: "small" | "middle" | "large";
  type?: "primary" | "default" | "dashed" | "link" | "text";
  className?: string;
}

const buttonVariants = {
  primary: "default",
  default: "outline",
  dashed: "outline",
  link: "link",
  text: "ghost",
} as const;

const buttonSizes = {
  small: "sm",
  middle: "default",
  large: "lg",
} as const;

const isValidReloadInterval = (value: number) => {
  if (!Number.isFinite(value)) return false;
  if (!Number.isInteger(value)) return false;
  return value >= 1 && value <= 168;
};

const formatDateTime = (dateTimeString: string | null, t: TFunction = i18n.t.bind(i18n)) => {
  if (!dateTimeString) return t("priceDataReload.never");
  const parsed = new Date(dateTimeString);
  return Number.isNaN(parsed.getTime()) ? dateTimeString : parsed.toLocaleString();
};

const CostMapProvenanceRows: React.FC<{ sourceInfo: CostMapSourceInfo }> = ({ sourceInfo }) => {
  const { t } = useTranslation();
  return (
    <>
      {sourceInfo.source_revision && (
        <div className="flex items-center justify-between gap-2 text-xs">
          <span className="text-muted-foreground">{t("priceDataReload.provenance.sourceRevision")}</span>
          <Tooltip>
            <TooltipTrigger render={<code className="font-mono" />}>
              {shortRevision(sourceInfo.source_revision)}
            </TooltipTrigger>
            <TooltipContent>{sourceInfo.source_revision}</TooltipContent>
          </Tooltip>
        </div>
      )}

      {sourceInfo.etag && (
        <div className="flex items-center justify-between gap-2 text-xs">
          <span className="text-muted-foreground">{t("priceDataReload.provenance.etag")}</span>
          <Tooltip>
            <TooltipTrigger render={<code className="max-w-60 truncate font-mono" />}>{sourceInfo.etag}</TooltipTrigger>
            <TooltipContent>{sourceInfo.etag}</TooltipContent>
          </Tooltip>
        </div>
      )}

      {sourceInfo.loaded_at && (
        <div className="flex items-center justify-between text-xs">
          <span className="text-muted-foreground">{t("priceDataReload.provenance.loadedAt")}</span>
          <span className="font-medium">{formatDateTime(sourceInfo.loaded_at, t)}</span>
        </div>
      )}

      {sourceInfo.loaded_at && (
        <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
          <Info className="size-3.5 shrink-0" />
          <span>{t("priceDataReload.provenance.workerNote")}</span>
        </div>
      )}
    </>
  );
};

const PriceDataReload: React.FC<PriceDataReloadProps> = ({
  accessToken,
  onReloadSuccess,
  buttonText = i18n.t("priceDataReload.defaultButtonText"),
  showIcon = true,
  size = "middle",
  type = "primary",
  className = "",
}) => {
  const { t } = useTranslation();
  const [isLoading, setIsLoading] = useState(false);
  const [isScheduling, setIsScheduling] = useState(false);
  const [isCancelling, setIsCancelling] = useState(false);
  const [showScheduleModal, setShowScheduleModal] = useState(false);
  const [hours, setHours] = useState<number | "">(6);
  const [reloadStatus, setReloadStatus] = useState<ReloadStatus | null>(null);
  const [sourceInfo, setSourceInfo] = useState<CostMapSourceInfo | null>(null);

  const fetchReloadStatus = async () => {
    if (!accessToken) return;

    try {
      const status = await getModelCostMapReloadStatus(accessToken);
      setReloadStatus(status);
    } catch (error) {
      console.error("Failed to fetch reload status:", error);
      setReloadStatus(EMPTY_RELOAD_STATUS);
    }
  };

  const fetchSourceInfo = async () => {
    if (!accessToken) return;

    try {
      setSourceInfo(await getModelCostMapSource(accessToken));
    } catch (error) {
      console.error("Failed to fetch cost map source info:", error);
    }
  };

  useEffect(() => {
    const initialRefresh = window.setTimeout(() => {
      fetchReloadStatus();
      fetchSourceInfo();
    }, 0);

    const interval = setInterval(() => {
      fetchReloadStatus();
      fetchSourceInfo();
    }, 30000);

    return () => {
      clearTimeout(initialRefresh);
      clearInterval(interval);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps -- refresh functions intentionally follow the current access token
  }, [accessToken]);

  const handleHardRefresh = async () => {
    if (!accessToken) {
      toast.fromError(t("priceDataReload.toasts.noAccessToken"));
      return;
    }

    setIsLoading(true);
    try {
      const response = await reloadModelCostMap(accessToken);

      if (response.status === "success") {
        toast.success(t("priceDataReload.toasts.reloadSuccess", { modelsCount: response.models_count || 0 }));
        onReloadSuccess?.();
        await fetchReloadStatus();
        await fetchSourceInfo();
      } else {
        toast.fromError(t("priceDataReload.toasts.reloadFailed"));
      }
    } catch (error) {
      console.error("Error reloading price data:", error);
      toast.fromError(t("priceDataReload.toasts.reloadFailedRetry"));
    } finally {
      setIsLoading(false);
    }
  };

  const handleScheduleReload = async () => {
    if (!accessToken) {
      toast.fromError(t("priceDataReload.toasts.noAccessToken"));
      return;
    }

    const intervalHours = Number(hours);
    if (!isValidReloadInterval(intervalHours)) {
      toast.fromError(t("priceDataReload.toasts.invalidHours"));
      return;
    }

    setIsScheduling(true);
    try {
      const response = await scheduleModelCostMapReload(accessToken, intervalHours);

      if (response.status === "success") {
        toast.success(t("priceDataReload.toasts.scheduleSuccess", { hours: intervalHours }));
        setShowScheduleModal(false);
        await fetchReloadStatus();
      } else {
        toast.fromError(t("priceDataReload.toasts.scheduleFailed"));
      }
    } catch (error) {
      console.error("Error scheduling reload:", error);
      toast.fromError(t("priceDataReload.toasts.scheduleFailedRetry"));
    } finally {
      setIsScheduling(false);
    }
  };

  const handleCancelReload = async () => {
    if (!accessToken) {
      toast.fromError(t("priceDataReload.toasts.noAccessToken"));
      return;
    }

    setIsCancelling(true);
    try {
      const response = await cancelModelCostMapReload(accessToken);

      if (response.status === "success") {
        toast.success(t("priceDataReload.toasts.cancelSuccess"));
        await fetchReloadStatus();
      } else {
        toast.fromError(t("priceDataReload.toasts.cancelFailed"));
      }
    } catch (error) {
      console.error("Error cancelling reload:", error);
      toast.fromError(t("priceDataReload.toasts.cancelFailedRetry"));
    } finally {
      setIsCancelling(false);
    }
  };

  const getStatusText = () => {
    if (!reloadStatus?.scheduled) return t("priceDataReload.status.notScheduled");
    if (!reloadStatus.last_run) return t("priceDataReload.status.ready");
    return t("priceDataReload.status.active");
  };

  return (
    <TooltipProvider>
      <div className={className}>
        <div className="mb-4 flex flex-wrap gap-3">
          <AlertDialog>
            <AlertDialogTrigger
              render={
                <Button
                  type="button"
                  variant={buttonVariants[type]}
                  size={buttonSizes[size]}
                  className={cn(type === "dashed" && "border-dashed")}
                  disabled={isLoading}
                />
              }
            >
              {isLoading ? (
                <LoaderCircle className="animate-spin" data-icon="inline-start" />
              ) : (
                showIcon && <RefreshCw data-icon="inline-start" />
              )}
              {buttonText}
            </AlertDialogTrigger>
            <AlertDialogContent>
              <AlertDialogHeader>
                <AlertDialogTitle>{t("priceDataReload.hardRefresh.title")}</AlertDialogTitle>
                <AlertDialogDescription>{t("priceDataReload.hardRefresh.description")}</AlertDialogDescription>
              </AlertDialogHeader>
              <AlertDialogFooter>
                <AlertDialogCancel>{t("priceDataReload.hardRefresh.no")}</AlertDialogCancel>
                <AlertDialogAction onClick={handleHardRefresh}>
                  {t("priceDataReload.hardRefresh.yes")}
                </AlertDialogAction>
              </AlertDialogFooter>
            </AlertDialogContent>
          </AlertDialog>

          {!reloadStatus?.scheduled ? (
            <Button type="button" variant="outline" size={buttonSizes[size]} onClick={() => setShowScheduleModal(true)}>
              <Clock3 data-icon="inline-start" />
              {t("priceDataReload.setUpPeriodicReload")}
            </Button>
          ) : (
            <Button
              type="button"
              variant="destructive"
              size={buttonSizes[size]}
              disabled={isCancelling}
              onClick={handleCancelReload}
            >
              {isCancelling ? (
                <LoaderCircle className="animate-spin" data-icon="inline-start" />
              ) : (
                <Ban data-icon="inline-start" />
              )}
              {t("priceDataReload.cancelPeriodicReload")}
            </Button>
          )}
        </div>

        {sourceInfo && (
          <Card size="sm" className="mb-3 bg-muted/30">
            <CardContent className="space-y-2">
              <div className="flex items-center gap-2">
                {sourceInfo.source === "remote" ? <Cloud className="size-4" /> : <Database className="size-4" />}
                <span className="text-sm font-medium">{t("priceDataReload.source.title")}</span>
                <Badge variant="secondary" className="ml-auto uppercase">
                  {sourceInfo.source === "remote"
                    ? t("priceDataReload.source.remote")
                    : t("priceDataReload.source.local")}
                </Badge>
              </div>

              <Separator />

              <div className="flex items-center justify-between text-xs">
                <span className="text-muted-foreground">{t("priceDataReload.source.modelsLoaded")}</span>
                <span className="font-medium">{sourceInfo.model_count.toLocaleString()}</span>
              </div>

              {sourceInfo.url && (
                <div className="flex items-start justify-between gap-2 text-xs">
                  <span className="shrink-0 text-muted-foreground">
                    {sourceInfo.source === "remote"
                      ? t("priceDataReload.source.loadedFrom")
                      : t("priceDataReload.source.attemptedUrl")}
                  </span>
                  <Tooltip>
                    <TooltipTrigger render={<span className="max-w-60 truncate text-primary" />}>
                      {sourceInfo.url}
                    </TooltipTrigger>
                    <TooltipContent>{sourceInfo.url}</TooltipContent>
                  </Tooltip>
                </div>
              )}

              <CostMapProvenanceRows sourceInfo={sourceInfo} />

              {sourceInfo.is_env_forced && (
                <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
                  <Info className="size-3.5 shrink-0" />
                  <span>
                    <Trans i18nKey="priceDataReload.source.envForced" components={{ code: <code /> }} />
                  </span>
                </div>
              )}

              {sourceInfo.fallback_reason && (
                <div className="flex items-start gap-1.5 rounded-md border border-destructive/30 bg-destructive/10 px-2 py-1.5 text-xs">
                  <TriangleAlert className="mt-0.5 size-3.5 shrink-0 text-destructive" />
                  <span>{t("priceDataReload.source.fallback", { reason: sourceInfo.fallback_reason })}</span>
                </div>
              )}
            </CardContent>
          </Card>
        )}

        {reloadStatus && (
          <Card size="sm" className="bg-muted/30">
            <CardContent className="space-y-2">
              {reloadStatus.scheduled ? (
                <Badge variant="secondary">
                  <Clock3 />
                  {t("priceDataReload.schedule.scheduledEvery", { hours: reloadStatus.interval_hours })}
                </Badge>
              ) : (
                <p className="text-sm text-muted-foreground">{t("priceDataReload.schedule.noneScheduled")}</p>
              )}

              <div className="flex items-center justify-between text-xs">
                <span className="text-muted-foreground">{t("priceDataReload.schedule.lastRun")}</span>
                <span>{formatDateTime(reloadStatus.last_run, t)}</span>
              </div>

              {reloadStatus.scheduled && (
                <>
                  {reloadStatus.next_run && (
                    <div className="flex items-center justify-between text-xs">
                      <span className="text-muted-foreground">{t("priceDataReload.schedule.nextRun")}</span>
                      <span>{formatDateTime(reloadStatus.next_run, t)}</span>
                    </div>
                  )}
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-muted-foreground">{t("priceDataReload.schedule.status")}</span>
                    <Badge variant="outline">{getStatusText()}</Badge>
                  </div>
                </>
              )}
            </CardContent>
          </Card>
        )}

        <Dialog open={showScheduleModal} onOpenChange={setShowScheduleModal}>
          <DialogContent className="max-h-[calc(100dvh-2rem)] overflow-y-auto">
            <DialogHeader>
              <DialogTitle>{t("priceDataReload.scheduleDialog.title")}</DialogTitle>
              <DialogDescription>{t("priceDataReload.scheduleDialog.description")}</DialogDescription>
            </DialogHeader>
            <div className="space-y-4">
              <p className="text-sm">{t("priceDataReload.scheduleDialog.intro")}</p>
              <InputGroup>
                <InputGroupInput
                  type="number"
                  aria-label={t("priceDataReload.scheduleDialog.intervalLabel")}
                  min={1}
                  max={168}
                  value={hours}
                  onChange={(event) => setHours(event.target.value === "" ? "" : Number(event.target.value))}
                />
                <InputGroupAddon align="inline-end">{t("priceDataReload.scheduleDialog.hoursUnit")}</InputGroupAddon>
              </InputGroup>
              <p className="text-sm text-muted-foreground">{t("priceDataReload.scheduleDialog.summary", { hours })}</p>
            </div>
            <DialogFooter>
              <Button type="button" variant="outline" onClick={() => setShowScheduleModal(false)}>
                {t("priceDataReload.scheduleDialog.cancel")}
              </Button>
              <Button type="button" disabled={isScheduling} onClick={handleScheduleReload}>
                {isScheduling && <LoaderCircle className="animate-spin" data-icon="inline-start" />}
                {t("priceDataReload.scheduleDialog.schedule")}
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>
    </TooltipProvider>
  );
};

export default PriceDataReload;
