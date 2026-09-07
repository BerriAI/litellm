import React, { useEffect, useState } from "react";
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
  model_count: number;
}

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

const PriceDataReload: React.FC<PriceDataReloadProps> = ({
  accessToken,
  onReloadSuccess,
  buttonText = "Reload Price Data",
  showIcon = true,
  size = "middle",
  type = "primary",
  className = "",
}) => {
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
      toast.fromError("No access token available");
      return;
    }

    setIsLoading(true);
    try {
      const response = await reloadModelCostMap(accessToken);

      if (response.status === "success") {
        toast.success(`Price data reloaded successfully! ${response.models_count || 0} models updated.`);
        onReloadSuccess?.();
        await fetchReloadStatus();
        await fetchSourceInfo();
      } else {
        toast.fromError("Failed to reload price data");
      }
    } catch (error) {
      console.error("Error reloading price data:", error);
      toast.fromError("Failed to reload price data. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleScheduleReload = async () => {
    if (!accessToken) {
      toast.fromError("No access token available");
      return;
    }

    const intervalHours = Number(hours);
    if (!isValidReloadInterval(intervalHours)) {
      toast.fromError("Hours must be a whole number between 1 and 168");
      return;
    }

    setIsScheduling(true);
    try {
      const response = await scheduleModelCostMapReload(accessToken, intervalHours);

      if (response.status === "success") {
        toast.success(`Periodic reload scheduled for every ${intervalHours} hours`);
        setShowScheduleModal(false);
        await fetchReloadStatus();
      } else {
        toast.fromError("Failed to schedule periodic reload");
      }
    } catch (error) {
      console.error("Error scheduling reload:", error);
      toast.fromError("Failed to schedule periodic reload. Please try again.");
    } finally {
      setIsScheduling(false);
    }
  };

  const handleCancelReload = async () => {
    if (!accessToken) {
      toast.fromError("No access token available");
      return;
    }

    setIsCancelling(true);
    try {
      const response = await cancelModelCostMapReload(accessToken);

      if (response.status === "success") {
        toast.success("Periodic reload cancelled successfully");
        await fetchReloadStatus();
      } else {
        toast.fromError("Failed to cancel periodic reload");
      }
    } catch (error) {
      console.error("Error cancelling reload:", error);
      toast.fromError("Failed to cancel periodic reload. Please try again.");
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
                <AlertDialogTitle>Hard Refresh Price Data</AlertDialogTitle>
                <AlertDialogDescription>
                  This will immediately fetch the latest pricing information from the remote source. Continue?
                </AlertDialogDescription>
              </AlertDialogHeader>
              <AlertDialogFooter>
                <AlertDialogCancel>No</AlertDialogCancel>
                <AlertDialogAction onClick={handleHardRefresh}>Yes</AlertDialogAction>
              </AlertDialogFooter>
            </AlertDialogContent>
          </AlertDialog>

          {!reloadStatus?.scheduled ? (
            <Button type="button" variant="outline" size={buttonSizes[size]} onClick={() => setShowScheduleModal(true)}>
              <Clock3 data-icon="inline-start" />
              Set Up Periodic Reload
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
              Cancel Periodic Reload
            </Button>
          )}
        </div>

        {sourceInfo && (
          <Card size="sm" className="mb-3 bg-muted/30">
            <CardContent className="space-y-2">
              <div className="flex items-center gap-2">
                {sourceInfo.source === "remote" ? <Cloud className="size-4" /> : <Database className="size-4" />}
                <span className="text-sm font-medium">Pricing Data Source</span>
                <Badge variant="secondary" className="ml-auto uppercase">
                  {sourceInfo.source === "remote" ? "Remote" : "Local"}
                </Badge>
              </div>

              <Separator />

              <div className="flex items-center justify-between text-xs">
                <span className="text-muted-foreground">Models loaded:</span>
                <span className="font-medium">{sourceInfo.model_count.toLocaleString()}</span>
              </div>

              {sourceInfo.url && (
                <div className="flex items-start justify-between gap-2 text-xs">
                  <span className="shrink-0 text-muted-foreground">
                    {sourceInfo.source === "remote" ? "Loaded from:" : "Attempted URL:"}
                  </span>
                  <Tooltip>
                    <TooltipTrigger render={<span className="max-w-60 truncate text-primary" />}>
                      {sourceInfo.url}
                    </TooltipTrigger>
                    <TooltipContent>{sourceInfo.url}</TooltipContent>
                  </Tooltip>
                </div>
              )}

              {sourceInfo.is_env_forced && (
                <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
                  <Info className="size-3.5 shrink-0" />
                  <span>
                    Local mode forced via <code>LITELLM_LOCAL_MODEL_COST_MAP=True</code>
                  </span>
                </div>
              )}

              {sourceInfo.fallback_reason && (
                <div className="flex items-start gap-1.5 rounded-md border border-destructive/30 bg-destructive/10 px-2 py-1.5 text-xs">
                  <TriangleAlert className="mt-0.5 size-3.5 shrink-0 text-destructive" />
                  <span>Fell back to local: {sourceInfo.fallback_reason}</span>
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
                  Scheduled every {reloadStatus.interval_hours} hours
                </Badge>
              ) : (
                <p className="text-sm text-muted-foreground">No periodic reload scheduled</p>
              )}

              <div className="flex items-center justify-between text-xs">
                <span className="text-muted-foreground">Last run:</span>
                <span>{formatDateTime(reloadStatus.last_run)}</span>
              </div>

              {reloadStatus.scheduled && (
                <>
                  {reloadStatus.next_run && (
                    <div className="flex items-center justify-between text-xs">
                      <span className="text-muted-foreground">Next run:</span>
                      <span>{formatDateTime(reloadStatus.next_run)}</span>
                    </div>
                  )}
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-muted-foreground">Status:</span>
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
              <DialogTitle>Set Up Periodic Reload</DialogTitle>
              <DialogDescription>
                Set how often LiteLLM should fetch the latest pricing data from the remote source.
              </DialogDescription>
            </DialogHeader>
            <div className="space-y-4">
              <p className="text-sm">Set up automatic reload of price data every:</p>
              <InputGroup>
                <InputGroupInput
                  type="number"
                  aria-label="Reload interval in hours"
                  min={1}
                  max={168}
                  value={hours}
                  onChange={(event) => setHours(event.target.value === "" ? "" : Number(event.target.value))}
                />
                <InputGroupAddon align="inline-end">hours</InputGroupAddon>
              </InputGroup>
              <p className="text-sm text-muted-foreground">
                This will automatically fetch the latest pricing data from the remote source every {hours} hours.
              </p>
            </div>
            <DialogFooter>
              <Button type="button" variant="outline" onClick={() => setShowScheduleModal(false)}>
                Cancel
              </Button>
              <Button type="button" disabled={isScheduling} onClick={handleScheduleReload}>
                {isScheduling && <LoaderCircle className="animate-spin" data-icon="inline-start" />}
                Schedule
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>
    </TooltipProvider>
  );
};

export default PriceDataReload;
