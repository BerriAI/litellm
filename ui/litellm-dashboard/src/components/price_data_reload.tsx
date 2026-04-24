import React, { useState, useEffect } from "react";
import {
  RefreshCcw as ReloadOutlined,
  Clock as ClockCircleOutlined,
  Ban as StopOutlined,
  Cloud as CloudOutlined,
  Database as DatabaseOutlined,
  Info as InfoCircleOutlined,
  AlertTriangle as WarningOutlined,
  Loader2,
} from "lucide-react";
import {
  reloadModelCostMap,
  scheduleModelCostMapReload,
  cancelModelCostMapReload,
  getModelCostMapReloadStatus,
  getModelCostMapSource,
} from "./networking";
import NotificationsManager from "./molecules/notifications_manager";
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
import { Card } from "@/components/ui/card";
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Separator } from "@/components/ui/separator";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

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
  model_count: number;
}

interface PriceDataReloadProps {
  accessToken: string;
  onReloadSuccess?: () => void;
  buttonText?: string;
  showIcon?: boolean;
  size?: "small" | "middle" | "large";
  type?: "primary" | "default" | "dashed" | "link" | "text";
  className?: string;
}

const PriceDataReload: React.FC<PriceDataReloadProps> = ({
  accessToken,
  onReloadSuccess,
  buttonText = "Reload Price Data",
  showIcon = true,
  // `size` and `type` are retained in props for API compatibility with
  // existing call-sites (they were antd Button props). They are not
  // currently wired into shadcn Button variants because visual intent
  // was always "primary indigo"; preserve that here.
  size: _size,
  type: _type,
  className = "",
}) => {
  const [isLoading, setIsLoading] = useState(false);
  const [isScheduling, setIsScheduling] = useState(false);
  const [isCancelling, setIsCancelling] = useState(false);
  const [showScheduleModal, setShowScheduleModal] = useState(false);
  const [hours, setHours] = useState<number>(6);
  const [reloadStatus, setReloadStatus] = useState<ReloadStatus | null>(null);
  const [sourceInfo, setSourceInfo] = useState<CostMapSourceInfo | null>(null);

  useEffect(() => {
    fetchReloadStatus();
    fetchSourceInfo();

    const interval = setInterval(() => {
      fetchReloadStatus();
      fetchSourceInfo();
    }, 30000);

    return () => clearInterval(interval);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [accessToken]);

  const fetchReloadStatus = async () => {
    if (!accessToken) return;
    try {
      const status = await getModelCostMapReloadStatus(accessToken);
      setReloadStatus(status);
    } catch (error) {
      console.error("Failed to fetch reload status:", error);
      setReloadStatus({
        scheduled: false,
        interval_hours: null,
        last_run: null,
        next_run: null,
      });
    }
  };

  const fetchSourceInfo = async () => {
    if (!accessToken) return;
    try {
      const info = await getModelCostMapSource(accessToken);
      setSourceInfo(info);
    } catch (error) {
      console.error("Failed to fetch cost map source info:", error);
    }
  };

  const handleHardRefresh = async () => {
    if (!accessToken) {
      NotificationsManager.fromBackend("No access token available");
      return;
    }

    setIsLoading(true);
    try {
      const response = await reloadModelCostMap(accessToken);
      if (response.status === "success") {
        NotificationsManager.success(
          `Price data reloaded successfully! ${response.models_count || 0} models updated.`,
        );
        onReloadSuccess?.();
        await fetchReloadStatus();
        await fetchSourceInfo();
      } else {
        NotificationsManager.fromBackend("Failed to reload price data");
      }
    } catch (error) {
      console.error("Error reloading price data:", error);
      NotificationsManager.fromBackend(
        "Failed to reload price data. Please try again.",
      );
    } finally {
      setIsLoading(false);
    }
  };

  const handleScheduleReload = async () => {
    if (!accessToken) {
      NotificationsManager.fromBackend("No access token available");
      return;
    }

    if (hours <= 0) {
      NotificationsManager.fromBackend("Hours must be greater than 0");
      return;
    }

    setIsScheduling(true);
    try {
      const response = await scheduleModelCostMapReload(accessToken, hours);
      if (response.status === "success") {
        NotificationsManager.success(
          `Periodic reload scheduled for every ${hours} hours`,
        );
        setShowScheduleModal(false);
        await fetchReloadStatus();
      } else {
        NotificationsManager.fromBackend("Failed to schedule periodic reload");
      }
    } catch (error) {
      console.error("Error scheduling reload:", error);
      NotificationsManager.fromBackend(
        "Failed to schedule periodic reload. Please try again.",
      );
    } finally {
      setIsScheduling(false);
    }
  };

  const handleCancelReload = async () => {
    if (!accessToken) {
      NotificationsManager.fromBackend("No access token available");
      return;
    }

    setIsCancelling(true);
    try {
      const response = await cancelModelCostMapReload(accessToken);
      if (response.status === "success") {
        NotificationsManager.success("Periodic reload cancelled successfully");
        await fetchReloadStatus();
      } else {
        NotificationsManager.fromBackend("Failed to cancel periodic reload");
      }
    } catch (error) {
      console.error("Error cancelling reload:", error);
      NotificationsManager.fromBackend(
        "Failed to cancel periodic reload. Please try again.",
      );
    } finally {
      setIsCancelling(false);
    }
  };

  const formatDateTime = (dateTimeString: string | null) => {
    if (!dateTimeString) return "Never";
    try {
      return new Date(dateTimeString).toLocaleString();
    } catch {
      return dateTimeString;
    }
  };

  const getStatusText = () => {
    if (!reloadStatus?.scheduled) return "Not scheduled";
    if (!reloadStatus.last_run) return "Ready";
    return "Active";
  };

  return (
    <div className={className}>
      {/* Action Buttons */}
      <div className="flex items-center gap-3 mb-4">
        {/* Hard Refresh Button with confirmation */}
        <AlertDialog>
          <AlertDialogTrigger asChild>
            <Button
              className="bg-indigo-500 hover:bg-indigo-600 text-white"
              disabled={isLoading}
            >
              {isLoading ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                showIcon && <ReloadOutlined className="h-4 w-4" />
              )}
              {buttonText}
            </Button>
          </AlertDialogTrigger>
          <AlertDialogContent>
            <AlertDialogHeader>
              <AlertDialogTitle>Hard Refresh Price Data</AlertDialogTitle>
              <AlertDialogDescription>
                This will immediately fetch the latest pricing information
                from the remote source. Continue?
              </AlertDialogDescription>
            </AlertDialogHeader>
            <AlertDialogFooter>
              <AlertDialogCancel>No</AlertDialogCancel>
              <AlertDialogAction
                onClick={handleHardRefresh}
                className="bg-indigo-500 hover:bg-indigo-600 text-white"
              >
                Yes
              </AlertDialogAction>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialog>

        {/* Periodic Reload Controls */}
        {!reloadStatus?.scheduled ? (
          <Button variant="outline" onClick={() => setShowScheduleModal(true)}>
            <ClockCircleOutlined className="h-4 w-4" />
            Set Up Periodic Reload
          </Button>
        ) : (
          <Button
            variant="outline"
            className="border-red-500 text-red-500 hover:bg-red-50 hover:text-red-600"
            disabled={isCancelling}
            onClick={handleCancelReload}
          >
            {isCancelling ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <StopOutlined className="h-4 w-4" />
            )}
            Cancel Periodic Reload
          </Button>
        )}
      </div>

      {/* Cost Map Source Info Card */}
      {sourceInfo && (
        <Card
          className={`p-3 mb-3 rounded-lg ${
            sourceInfo.source === "remote"
              ? "bg-blue-50 border-blue-200"
              : "bg-orange-50 border-orange-200"
          }`}
        >
          <div className="flex flex-col gap-2 w-full">
            {/* Header row */}
            <div className="flex items-center gap-2">
              {sourceInfo.source === "remote" ? (
                <CloudOutlined className="h-4 w-4 text-blue-600" />
              ) : (
                <DatabaseOutlined className="h-4 w-4 text-orange-500" />
              )}
              <span className="font-semibold text-[13px]">
                Pricing Data Source
              </span>
              <Badge
                className={`ml-auto font-semibold uppercase text-[11px] ${
                  sourceInfo.source === "remote"
                    ? "bg-blue-100 text-blue-700"
                    : "bg-orange-100 text-orange-700"
                }`}
              >
                {sourceInfo.source === "remote" ? "Remote" : "Local"}
              </Badge>
            </div>

            <Separator className="my-1" />

            {/* Model count */}
            <div className="flex items-center justify-between">
              <span className="text-muted-foreground text-xs">
                Models loaded:
              </span>
              <span className="font-semibold text-xs">
                {sourceInfo.model_count.toLocaleString()}
              </span>
            </div>

            {/* URL */}
            {sourceInfo.url && (
              <div className="flex items-start justify-between gap-2">
                <span className="text-muted-foreground text-xs whitespace-nowrap">
                  {sourceInfo.source === "remote"
                    ? "Loaded from:"
                    : "Attempted URL:"}
                </span>
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <span className="text-[11px] max-w-[240px] truncate block text-blue-600 cursor-default">
                        {sourceInfo.url}
                      </span>
                    </TooltipTrigger>
                    <TooltipContent>{sourceInfo.url}</TooltipContent>
                  </Tooltip>
                </TooltipProvider>
              </div>
            )}

            {/* Env forced notice */}
            {sourceInfo.is_env_forced && (
              <div className="flex items-center gap-1.5 mt-0.5">
                <InfoCircleOutlined className="h-3 w-3 text-orange-500" />
                <span className="text-muted-foreground text-[11px]">
                  Local mode forced via{" "}
                  <code>LITELLM_LOCAL_MODEL_COST_MAP=True</code>
                </span>
              </div>
            )}

            {/* Fallback reason */}
            {sourceInfo.fallback_reason && (
              <div className="flex items-start gap-1.5 bg-orange-50 border border-orange-200 rounded px-2 py-1 mt-0.5">
                <WarningOutlined className="h-3 w-3 text-orange-500 mt-0.5" />
                <span className="text-[11px] text-orange-800">
                  Fell back to local: {sourceInfo.fallback_reason}
                </span>
              </div>
            )}
          </div>
        </Card>
      )}

      {/* Reload Schedule Status Card */}
      {reloadStatus && (
        <Card className="p-3 bg-muted border-border rounded-lg">
          <div className="flex flex-col gap-2 w-full">
            {reloadStatus.scheduled ? (
              <div>
                <Badge className="bg-green-100 text-green-700 gap-1 inline-flex items-center">
                  <ClockCircleOutlined className="h-3 w-3" />
                  Scheduled every {reloadStatus.interval_hours} hours
                </Badge>
              </div>
            ) : (
              <span className="text-muted-foreground text-sm">
                No periodic reload scheduled
              </span>
            )}

            <div className="flex items-center justify-between">
              <span className="text-muted-foreground text-xs">Last run:</span>
              <span className="text-xs">
                {formatDateTime(reloadStatus.last_run)}
              </span>
            </div>

            {reloadStatus.scheduled && (
              <>
                {reloadStatus.next_run && (
                  <div className="flex items-center justify-between">
                    <span className="text-muted-foreground text-xs">
                      Next run:
                    </span>
                    <span className="text-xs">
                      {formatDateTime(reloadStatus.next_run)}
                    </span>
                  </div>
                )}
                <div className="flex items-center justify-between">
                  <span className="text-muted-foreground text-xs">
                    Status:
                  </span>
                  <Badge
                    className={`${
                      reloadStatus.last_run
                        ? "bg-green-100 text-green-700"
                        : "bg-blue-100 text-blue-700"
                    }`}
                  >
                    {getStatusText()}
                  </Badge>
                </div>
              </>
            )}
          </div>
        </Card>
      )}

      {/* Schedule Modal */}
      <Dialog open={showScheduleModal} onOpenChange={setShowScheduleModal}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Set Up Periodic Reload</DialogTitle>
          </DialogHeader>
          <div className="mb-4">
            Set up automatic reload of price data every:
          </div>
          <div className="mb-4 flex items-center gap-2">
            <Input
              type="number"
              min={1}
              max={168}
              value={hours}
              onChange={(e) => setHours(Number(e.target.value) || 6)}
              className="w-full"
            />
            <span className="text-sm text-muted-foreground whitespace-nowrap">
              hours
            </span>
          </div>
          <div>
            <span className="text-muted-foreground text-sm">
              This will automatically fetch the latest pricing data from the
              remote source every {hours} hours.
            </span>
          </div>
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setShowScheduleModal(false)}
            >
              Cancel
            </Button>
            <Button
              className="bg-indigo-500 hover:bg-indigo-600 text-white"
              disabled={isScheduling}
              onClick={handleScheduleReload}
            >
              {isScheduling && <Loader2 className="h-4 w-4 animate-spin" />}
              Schedule
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
};

export default PriceDataReload;
