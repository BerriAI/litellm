import { OnChangeFn, PaginationState, RowSelectionState } from "@tanstack/react-table";
import type { TFunction } from "i18next";
import React, { useCallback, useEffect, useMemo, useState } from "react";
import { useTranslation } from "react-i18next";

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import i18n from "@/i18n/i18n";
import { errorPatterns } from "@/utils/errorPatterns";

import { Team } from "../key_team_helpers/key_list";
import { individualModelHealthCheckCall, latestHealthChecksCall } from "../networking";
import { HealthChecksTable } from "./HealthChecksTable";
import type { HealthCheckData, HealthStatus } from "./HealthChecksTableColumns";

interface LatestHealthCheck {
  status?: string;
  checked_at?: string | null;
  error_message?: string | null;
}

const STATUS_TO_ERROR: Record<string, string> = {
  "400": "BadRequestError",
  "401": "AuthenticationError",
  "403": "ForbiddenError",
  "404": "NotFoundError",
  "408": "TimeoutError",
  "429": "RateLimitError",
  "500": "InternalServerError",
  "502": "BadGatewayError",
  "503": "ServiceUnavailableError",
  "504": "GatewayTimeoutError",
};

const ERROR_TO_STATUS: Record<string, string> = {
  AuthenticationError: "401",
  RateLimitError: "429",
  BadRequestError: "400",
  InternalServerError: "500",
  TimeoutError: "408",
  NotFoundError: "404",
  ForbiddenError: "403",
  ServiceUnavailableError: "503",
  BadGatewayError: "502",
  GatewayTimeoutError: "504",
  ContentPolicyViolationError: "400",
};

const KEYWORD_ERRORS: ReadonlyArray<{ pattern: RegExp; label: string }> = [
  { pattern: /missing.*api.*key|invalid.*key|unauthorized/i, label: "AuthenticationError: 401" },
  { pattern: /rate.*limit|too.*many.*requests/i, label: "RateLimitError: 429" },
  { pattern: /timeout|timed.*out/i, label: "TimeoutError: 408" },
  { pattern: /not.*found/i, label: "NotFoundError: 404" },
  { pattern: /forbidden|access.*denied/i, label: "ForbiddenError: 403" },
  { pattern: /internal.*server.*error/i, label: "InternalServerError: 500" },
];

const truncate = (value: string): string => (value.length > 100 ? `${value.substring(0, 97)}...` : value);

// Helper function to extract meaningful error information
const extractMeaningfulError = (error: unknown, t: TFunction = i18n.t.bind(i18n)): string => {
  if (!error) return t("modelDashboard.healthCheckComponent.healthCheckFailed");

  const errorStr = typeof error === "string" ? error : JSON.stringify(error);

  // First, look for explicit "ErrorType: StatusCode" patterns
  const directPatternMatch = errorStr.match(/(\w+Error):\s*(\d{3})/i);
  if (directPatternMatch) {
    return `${directPatternMatch[1]}: ${directPatternMatch[2]}`;
  }

  // Look for error types and status codes separately, then combine them
  const errorTypeMatch = errorStr.match(
    /(AuthenticationError|RateLimitError|BadRequestError|InternalServerError|TimeoutError|NotFoundError|ForbiddenError|ServiceUnavailableError|BadGatewayError|ContentPolicyViolationError|\w+Error)/i,
  );
  const statusCodeMatch = errorStr.match(/\b(400|401|403|404|408|429|500|502|503|504)\b/);

  if (errorTypeMatch && statusCodeMatch) {
    return `${errorTypeMatch[1]}: ${statusCodeMatch[1]}`;
  }

  // If we have a status code but no clear error type, map it
  if (statusCodeMatch) {
    const statusCode = statusCodeMatch[1];
    return `${STATUS_TO_ERROR[statusCode]}: ${statusCode}`;
  }

  // If we have an error type but no status code, map error type to expected status code
  if (errorTypeMatch) {
    const errorType = errorTypeMatch[1];
    const mappedStatus = ERROR_TO_STATUS[errorType];
    if (mappedStatus) {
      return `${errorType}: ${mappedStatus}`;
    }
    return errorType;
  }

  // Check for specific error patterns from errorPatterns
  for (const { pattern, replacement } of errorPatterns) {
    if (pattern.test(errorStr)) {
      return t(replacement);
    }
  }

  // Look for common error keywords and provide meaningful names with status codes
  for (const { pattern, label } of KEYWORD_ERRORS) {
    if (pattern.test(errorStr)) {
      return label;
    }
  }

  // Fallback: clean up the error string and return first meaningful part
  const cleaned = errorStr
    .replace(/[\n\r]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();

  // Try to get first meaningful sentence or phrase
  const firstSentence = cleaned.split(/[.!?]/)[0]?.trim();
  if (firstSentence && firstSentence.length > 0) {
    return truncate(firstSentence);
  }

  return truncate(cleaned);
};

const checkedAtOr = (checkedAt: string | null | undefined, fallback: string): string => checkedAt || fallback;

const lastSuccessAtOr = (checkData: LatestHealthCheck, fallback: string): string =>
  checkData.status === "healthy" ? checkedAtOr(checkData.checked_at, fallback) : fallback;

interface HealthCheckComponentProps {
  accessToken: string | null;
  modelData: any;
  all_models_on_proxy: string[];
  getDisplayModelName: (model: any) => string;
  setSelectedModelId?: (modelId: string) => void;
  teams?: Team[] | null;
  isLoading?: boolean;
  pagination: PaginationState;
  onPaginationChange: OnChangeFn<PaginationState>;
  rowCount: number;
}

const HealthCheckComponent: React.FC<HealthCheckComponentProps> = ({
  accessToken,
  modelData,
  all_models_on_proxy,
  getDisplayModelName,
  setSelectedModelId,
  teams,
  isLoading = false,
  pagination,
  onPaginationChange,
  rowCount,
}) => {
  const { t } = useTranslation();
  const [modelHealthStatuses, setModelHealthStatuses] = useState<{ [key: string]: HealthStatus }>({});
  const [rowSelection, setRowSelection] = useState<RowSelectionState>({});
  const [errorModalVisible, setErrorModalVisible] = useState(false);
  const [selectedErrorDetails, setSelectedErrorDetails] = useState<{
    modelName: string;
    cleanedError: string;
    fullError: string;
  } | null>(null);
  const [successModalVisible, setSuccessModalVisible] = useState(false);
  const [selectedSuccessDetails, setSelectedSuccessDetails] = useState<{
    modelName: string;
    response: unknown;
  } | null>(null);

  // Initialize health statuses on component mount (keyed by model id)
  useEffect(() => {
    if (!accessToken || !modelData?.data) return;

    const initializeHealthStatuses = async () => {
      const healthStatusMap: { [key: string]: HealthStatus } = {};

      // Initialize all models with default state using model ids
      modelData.data.forEach((model: any) => {
        const modelId = model.model_info?.id;
        if (modelId) {
          healthStatusMap[modelId] = {
            status: "none",
            lastCheck: "None",
            lastSuccess: "None",
            loading: false,
            error: undefined,
            fullError: undefined,
            successResponse: undefined,
          };
        }
      });

      try {
        const latestHealthChecks = await latestHealthChecksCall(accessToken);

        // Override with actual database data if it exists (latest_health_checks is keyed by model_id)
        if (
          latestHealthChecks &&
          latestHealthChecks.latest_health_checks &&
          typeof latestHealthChecks.latest_health_checks === "object"
        ) {
          Object.entries(latestHealthChecks.latest_health_checks).forEach(([modelId, rawCheck]) => {
            if (!rawCheck) return;
            const checkData = rawCheck as LatestHealthCheck;

            // Key is model_id from the backend (guaranteed by DB schema)
            const modelExists = modelData.data.some((m: any) => m.model_info?.id === modelId);
            if (!modelExists) return;

            const fullError = checkData.error_message || undefined;

            healthStatusMap[modelId] = {
              status: checkData.status || "unknown",
              lastCheck: checkedAtOr(checkData.checked_at, "None"),
              lastSuccess: lastSuccessAtOr(checkData, "None"),
              loading: false,
              error: fullError ? extractMeaningfulError(fullError) : undefined,
              fullError: fullError,
              successResponse: checkData.status === "healthy" ? checkData : undefined,
            };
          });
        }
      } catch (healthError) {
        console.warn("Failed to load health check history (using default states):", healthError);
      }

      setModelHealthStatuses(healthStatusMap);
    };

    initializeHealthStatuses();
  }, [accessToken, modelData]);

  const runIndividualHealthCheck = useCallback(
    async (modelId: string) => {
      if (!accessToken) return;

      setModelHealthStatuses((prev) => ({
        ...prev,
        [modelId]: {
          ...prev[modelId],
          loading: true,
          status: "checking",
        },
      }));

      try {
        const response = await individualModelHealthCheckCall(accessToken, modelId);
        const currentTime = new Date().toISOString();

        if (response.unhealthy_count > 0 && response.unhealthy_endpoints && response.unhealthy_endpoints.length > 0) {
          const rawError =
            response.unhealthy_endpoints[0]?.error || i18n.t("modelDashboard.healthCheckComponent.healthCheckFailed");
          const errorMessage = extractMeaningfulError(rawError);
          setModelHealthStatuses((prev) => ({
            ...prev,
            [modelId]: {
              status: "unhealthy",
              lastCheck: currentTime,
              lastSuccess: prev[modelId]?.lastSuccess || "None",
              loading: false,
              error: errorMessage,
              fullError: rawError,
            },
          }));
        } else {
          setModelHealthStatuses((prev) => ({
            ...prev,
            [modelId]: {
              status: "healthy",
              lastCheck: currentTime,
              lastSuccess: currentTime,
              loading: false,
              successResponse: response,
            },
          }));
        }

        try {
          const latestHealthChecks = await latestHealthChecksCall(accessToken);
          const checkData = latestHealthChecks.latest_health_checks?.[modelId] as LatestHealthCheck | undefined;

          if (checkData) {
            const fullError = checkData.error_message || undefined;
            setModelHealthStatuses((prev) => ({
              ...prev,
              [modelId]: {
                status: checkData.status || prev[modelId]?.status || "unknown",
                lastCheck: checkedAtOr(checkData.checked_at, prev[modelId]?.lastCheck || "None"),
                lastSuccess: lastSuccessAtOr(checkData, prev[modelId]?.lastSuccess || "None"),
                loading: false,
                error: fullError ? extractMeaningfulError(fullError) : prev[modelId]?.error,
                fullError: fullError || prev[modelId]?.fullError,
                successResponse: checkData.status === "healthy" ? checkData : prev[modelId]?.successResponse,
              },
            }));
          }
        } catch (dbError) {}
      } catch (error) {
        const currentTime = new Date().toISOString();
        const rawError = error instanceof Error ? error.message : String(error);
        const errorMessage = extractMeaningfulError(rawError);
        setModelHealthStatuses((prev) => ({
          ...prev,
          [modelId]: {
            status: "unhealthy",
            lastCheck: currentTime,
            lastSuccess: prev[modelId]?.lastSuccess || "None",
            loading: false,
            error: errorMessage,
            fullError: rawError,
          },
        }));
      }
    },
    [accessToken],
  );

  const selectedModelIds = useMemo(
    () => Object.keys(rowSelection).filter((modelId) => rowSelection[modelId]),
    [rowSelection],
  );

  const runAllHealthChecks = async () => {
    const modelsToCheck = selectedModelIds.length > 0 ? selectedModelIds : all_models_on_proxy;

    const loadingStatuses = modelsToCheck.reduce(
      (acc, modelId) => {
        acc[modelId] = {
          ...modelHealthStatuses[modelId],
          loading: true,
          status: "checking",
        };
        return acc;
      },
      {} as typeof modelHealthStatuses,
    );

    setModelHealthStatuses((prev) => ({ ...prev, ...loadingStatuses }));

    const healthCheckPromises = modelsToCheck.map(async (modelId) => {
      if (!accessToken) return;

      try {
        const response = await individualModelHealthCheckCall(accessToken, modelId);

        const currentTime = new Date().toISOString();
        if (response.unhealthy_count > 0 && response.unhealthy_endpoints && response.unhealthy_endpoints.length > 0) {
          const rawError =
            response.unhealthy_endpoints[0]?.error || t("modelDashboard.healthCheckComponent.healthCheckFailed");
          const errorMessage = extractMeaningfulError(rawError, t);
          setModelHealthStatuses((prev) => ({
            ...prev,
            [modelId]: {
              status: "unhealthy",
              lastCheck: currentTime,
              lastSuccess: prev[modelId]?.lastSuccess || "None",
              loading: false,
              error: errorMessage,
              fullError: rawError,
            },
          }));
        } else {
          setModelHealthStatuses((prev) => ({
            ...prev,
            [modelId]: {
              status: "healthy",
              lastCheck: currentTime,
              lastSuccess: currentTime,
              loading: false,
              successResponse: response,
            },
          }));
        }
      } catch (error) {
        console.error(`Health check failed for model id ${modelId}:`, error);
        const currentTime = new Date().toISOString();
        const rawError = error instanceof Error ? error.message : String(error);
        const errorMessage = extractMeaningfulError(rawError, t);
        setModelHealthStatuses((prev) => ({
          ...prev,
          [modelId]: {
            status: "unhealthy",
            lastCheck: currentTime,
            lastSuccess: prev[modelId]?.lastSuccess || "None",
            loading: false,
            error: errorMessage,
            fullError: rawError,
          },
        }));
      }
    });

    await Promise.allSettled(healthCheckPromises);

    try {
      if (!accessToken) return;
      const latestHealthChecks = await latestHealthChecksCall(accessToken);

      if (latestHealthChecks.latest_health_checks) {
        Object.entries(latestHealthChecks.latest_health_checks).forEach(([modelId, rawCheck]) => {
          if (!modelsToCheck.includes(modelId) || !rawCheck) return;
          const checkData = rawCheck as LatestHealthCheck;
          const fullError = checkData.error_message || undefined;

          setModelHealthStatuses((prev) => {
            const currentStatus = prev[modelId];
            return {
              ...prev,
              [modelId]: {
                status: checkData.status || currentStatus?.status || "unknown",
                lastCheck: checkedAtOr(checkData.checked_at, currentStatus?.lastCheck || "None"),
                lastSuccess: lastSuccessAtOr(checkData, currentStatus?.lastSuccess || "None"),
                loading: false,
                error: fullError ? extractMeaningfulError(fullError, t) : currentStatus?.error,
                fullError: fullError || currentStatus?.fullError,
                successResponse: checkData.status === "healthy" ? checkData : currentStatus?.successResponse,
              },
            };
          });
        });
      }
    } catch (dbError) {
      console.warn("Failed to fetch updated health statuses from database (non-critical):", dbError);
    }
  };

  // Changing the page swaps the underlying rows, so a carried-over selection would
  // point at models that are no longer on screen.
  const handlePaginationChange = useCallback<OnChangeFn<PaginationState>>(
    (updaterOrValue) => {
      setRowSelection({});
      setModelHealthStatuses({});
      onPaginationChange(updaterOrValue);
    },
    [onPaginationChange],
  );

  const showErrorModal = useCallback((modelName: string, cleanedError: string, fullError: string) => {
    setSelectedErrorDetails({ modelName, cleanedError, fullError });
    setErrorModalVisible(true);
  }, []);

  const closeErrorModal = () => {
    setErrorModalVisible(false);
    setSelectedErrorDetails(null);
  };

  const showSuccessModal = useCallback((modelName: string, response: unknown) => {
    setSelectedSuccessDetails({ modelName, response });
    setSuccessModalVisible(true);
  }, []);

  const closeSuccessModal = () => {
    setSuccessModalVisible(false);
    setSelectedSuccessDetails(null);
  };

  const healthTableData = useMemo<HealthCheckData[]>(
    () =>
      (modelData?.data ?? []).map((model: any) => {
        const modelId = model.model_info?.id;
        const healthStatus = modelId ? modelHealthStatuses[modelId] : null;
        const status = healthStatus || {
          status: "none",
          lastCheck: "None",
          loading: false,
        };
        return {
          model_name: model.model_name,
          model_info: model.model_info,
          provider: model.provider,
          litellm_model_name: model.litellm_model_name,
          health_status: status.status,
          last_check: status.lastCheck,
          last_success: status.lastSuccess || "None",
          health_loading: status.loading,
          health_error: status.error,
          health_full_error: status.fullError,
        };
      }),
    [modelData, modelHealthStatuses],
  );

  const isPartialSelection = selectedModelIds.length > 0 && selectedModelIds.length < all_models_on_proxy.length;
  const anyCheckRunning = Object.values(modelHealthStatuses).some((status) => status.loading);

  return (
    <div>
      <div className="mb-6">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-lg font-semibold text-foreground">{t("modelDashboard.healthCheckComponent.title")}</h2>
            <p className="mt-1 text-sm text-muted-foreground">{t("modelDashboard.healthCheckComponent.description")}</p>
          </div>
          <div className="flex items-center gap-3">
            {selectedModelIds.length > 0 && (
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setRowSelection({})}
                data-testid="clear-health-selection"
              >
                {t("modelDashboard.healthCheckComponent.clearSelection")}
              </Button>
            )}
            <Button
              variant="outline"
              size="sm"
              onClick={runAllHealthChecks}
              disabled={anyCheckRunning}
              data-testid="run-health-checks"
            >
              {isPartialSelection
                ? t("modelDashboard.healthCheckComponent.runSelectedChecks")
                : t("modelDashboard.healthCheckComponent.runAllChecks")}
            </Button>
          </div>
        </div>
      </div>

      <HealthChecksTable
        data={healthTableData}
        rowCount={rowCount}
        isLoading={isLoading}
        pagination={pagination}
        onPaginationChange={handlePaginationChange}
        rowSelection={rowSelection}
        onRowSelectionChange={setRowSelection}
        modelHealthStatuses={modelHealthStatuses}
        getDisplayModelName={getDisplayModelName}
        onRunHealthCheck={runIndividualHealthCheck}
        onShowError={showErrorModal}
        onShowSuccess={showSuccessModal}
        onSelectModel={setSelectedModelId}
        teams={teams}
      />

      <Dialog
        open={errorModalVisible}
        onOpenChange={(open) => {
          if (!open) closeErrorModal();
        }}
      >
        <DialogContent className="max-h-[calc(100dvh-2rem)] overflow-y-auto sm:max-w-3xl">
          <DialogHeader>
            <DialogTitle>
              {selectedErrorDetails
                ? t("modelDashboard.healthCheckComponent.errorDialog.titleWithModel", {
                    modelName: selectedErrorDetails.modelName,
                  })
                : t("modelDashboard.healthCheckComponent.errorDialog.title")}
            </DialogTitle>
            <DialogDescription>{t("modelDashboard.healthCheckComponent.errorDialog.description")}</DialogDescription>
          </DialogHeader>
          {selectedErrorDetails && (
            <div className="space-y-4">
              <div>
                <span className="font-medium">{t("modelDashboard.healthCheckComponent.errorDialog.errorLabel")}</span>
                <div className="mt-2 rounded-md border border-destructive/30 bg-destructive/10 p-3">
                  <span className="text-destructive">{selectedErrorDetails.cleanedError}</span>
                </div>
              </div>

              <div>
                <span className="font-medium">
                  {t("modelDashboard.healthCheckComponent.errorDialog.fullErrorLabel")}
                </span>
                <div className="mt-2 max-h-96 overflow-y-auto rounded-md border bg-muted/50 p-3">
                  <pre className="whitespace-pre-wrap text-sm text-foreground">{selectedErrorDetails.fullError}</pre>
                </div>
              </div>
            </div>
          )}
          <DialogFooter>
            <Button type="button" variant="outline" onClick={closeErrorModal}>
              {t("modelDashboard.healthCheckComponent.errorDialog.close")}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog
        open={successModalVisible}
        onOpenChange={(open) => {
          if (!open) closeSuccessModal();
        }}
      >
        <DialogContent className="max-h-[calc(100dvh-2rem)] overflow-y-auto sm:max-w-3xl">
          <DialogHeader>
            <DialogTitle>
              {selectedSuccessDetails
                ? t("modelDashboard.healthCheckComponent.successDialog.titleWithModel", {
                    modelName: selectedSuccessDetails.modelName,
                  })
                : t("modelDashboard.healthCheckComponent.successDialog.title")}
            </DialogTitle>
            <DialogDescription>{t("modelDashboard.healthCheckComponent.successDialog.description")}</DialogDescription>
          </DialogHeader>
          {selectedSuccessDetails && (
            <div className="space-y-4">
              <div>
                <span className="font-medium">
                  {t("modelDashboard.healthCheckComponent.successDialog.statusLabel")}
                </span>
                <div className="mt-2 rounded-md border border-primary/30 bg-primary/5 p-3">
                  <span className="text-foreground">
                    {t("modelDashboard.healthCheckComponent.successDialog.passed")}
                  </span>
                </div>
              </div>

              <div>
                <span className="font-medium">
                  {t("modelDashboard.healthCheckComponent.successDialog.responseLabel")}
                </span>
                <div className="mt-2 max-h-96 overflow-y-auto rounded-md border bg-muted/50 p-3">
                  <pre className="whitespace-pre-wrap text-sm text-foreground">
                    {JSON.stringify(selectedSuccessDetails.response, null, 2)}
                  </pre>
                </div>
              </div>
            </div>
          )}
          <DialogFooter>
            <Button type="button" variant="outline" onClick={closeSuccessModal}>
              {t("modelDashboard.healthCheckComponent.successDialog.close")}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
};

export default HealthCheckComponent;
