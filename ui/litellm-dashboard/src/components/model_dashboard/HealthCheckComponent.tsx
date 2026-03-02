import React, { useState, useEffect, useRef } from "react";
import { Title, Text, Button, Badge } from "@tremor/react";
import { Modal } from "antd";
import { Button as AntdButton } from "antd";
import { ModelDataTable } from "./table";
import { healthCheckColumns } from "./health_check_columns";
import { errorPatterns } from "@/utils/errorPatterns";
import { individualModelHealthCheckCall, latestHealthChecksCall } from "../networking";
import { Table as TableInstance } from "@tanstack/react-table";
import { Team } from "../key_team_helpers/key_list";

interface HealthStatus {
  status: string;
  lastCheck: string;
  lastSuccess?: string;
  loading: boolean;
  error?: string;
  fullError?: string;
  successResponse?: any;
}

interface HealthCheckComponentProps {
  accessToken: string | null;
  modelData: any;
  all_models_on_proxy: string[];
  getDisplayModelName: (model: any) => string;
  setSelectedModelId?: (modelId: string) => void;
  teams?: Team[] | null;
}

const HealthCheckComponent: React.FC<HealthCheckComponentProps> = ({
  accessToken,
  modelData,
  all_models_on_proxy,
  getDisplayModelName,
  setSelectedModelId,
  teams,
}) => {
  const [modelHealthStatuses, setModelHealthStatuses] = useState<{ [key: string]: HealthStatus }>({});
  const [selectedModelsForHealth, setSelectedModelsForHealth] = useState<string[]>([]);
  const [allModelsSelected, setAllModelsSelected] = useState<boolean>(false);
  const [errorModalVisible, setErrorModalVisible] = useState(false);
  const [selectedErrorDetails, setSelectedErrorDetails] = useState<{
    modelName: string;
    cleanedError: string;
    fullError: string;
  } | null>(null);
  const [successModalVisible, setSuccessModalVisible] = useState(false);
  const [selectedSuccessDetails, setSelectedSuccessDetails] = useState<{
    modelName: string;
    response: any;
  } | null>(null);

  const healthTableRef = useRef<TableInstance<any>>(null);

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
          Object.entries(latestHealthChecks.latest_health_checks).forEach(([modelId, checkData]: [string, any]) => {
            if (!checkData) return;

            // Key is model_id from the backend (guaranteed by DB schema)
            const modelExists = modelData.data.some((m: any) => m.model_info?.id === modelId);
            if (!modelExists) return;

            const fullError = checkData.error_message || undefined;

            healthStatusMap[modelId] = {
                status: checkData.status || "unknown",
                lastCheck: checkData.checked_at ? new Date(checkData.checked_at).toLocaleString() : "None",
                lastSuccess:
                  checkData.status === "healthy"
                    ? checkData.checked_at
                      ? new Date(checkData.checked_at).toLocaleString()
                      : "None"
                    : "None",
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

  // Helper function to extract meaningful error information
  const extractMeaningfulError = (error: any): string => {
    if (!error) return "Health check failed";

    let errorStr = typeof error === "string" ? error : JSON.stringify(error);

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
      const statusToError: { [key: string]: string } = {
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
      return `${statusToError[statusCode]}: ${statusCode}`;
    }

    // If we have an error type but no status code, map error type to expected status code
    if (errorTypeMatch) {
      const errorType = errorTypeMatch[1];
      const errorToStatus: { [key: string]: string } = {
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

      const mappedStatus = errorToStatus[errorType];
      if (mappedStatus) {
        return `${errorType}: ${mappedStatus}`;
      }
      return errorType;
    }

    // Check for specific error patterns from errorPatterns
    for (const { pattern, replacement } of errorPatterns) {
      if (pattern.test(errorStr)) {
        return replacement;
      }
    }

    // Look for common error keywords and provide meaningful names with status codes
    if (/missing.*api.*key|invalid.*key|unauthorized/i.test(errorStr)) {
      return "AuthenticationError: 401";
    }
    if (/rate.*limit|too.*many.*requests/i.test(errorStr)) {
      return "RateLimitError: 429";
    }
    if (/timeout|timed.*out/i.test(errorStr)) {
      return "TimeoutError: 408";
    }
    if (/not.*found/i.test(errorStr)) {
      return "NotFoundError: 404";
    }
    if (/forbidden|access.*denied/i.test(errorStr)) {
      return "ForbiddenError: 403";
    }
    if (/internal.*server.*error/i.test(errorStr)) {
      return "InternalServerError: 500";
    }

    // Fallback: clean up the error string and return first meaningful part
    const cleaned = errorStr
      .replace(/[\n\r]+/g, " ")
      .replace(/\s+/g, " ")
      .trim();

    // Try to get first meaningful sentence or phrase
    const sentences = cleaned.split(/[.!?]/);
    const firstSentence = sentences[0]?.trim();

    if (firstSentence && firstSentence.length > 0) {
      return firstSentence.length > 100 ? firstSentence.substring(0, 97) + "..." : firstSentence;
    }

    return cleaned.length > 100 ? cleaned.substring(0, 97) + "..." : cleaned;
  };

  const runIndividualHealthCheck = async (modelId: string) => {
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
      const currentTime = new Date().toLocaleString();

      if (response.unhealthy_count > 0 && response.unhealthy_endpoints && response.unhealthy_endpoints.length > 0) {
        const rawError = response.unhealthy_endpoints[0]?.error || "Health check failed";
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
        const checkData = latestHealthChecks.latest_health_checks?.[modelId];

        if (checkData) {
          const fullError = checkData.error_message || undefined;
          setModelHealthStatuses((prev) => ({
            ...prev,
            [modelId]: {
              status: checkData.status || prev[modelId]?.status || "unknown",
              lastCheck: checkData.checked_at
                ? new Date(checkData.checked_at).toLocaleString()
                : prev[modelId]?.lastCheck || "None",
              lastSuccess:
                checkData.status === "healthy"
                  ? checkData.checked_at
                    ? new Date(checkData.checked_at).toLocaleString()
                    : prev[modelId]?.lastSuccess || "None"
                  : prev[modelId]?.lastSuccess || "None",
              loading: false,
              error: fullError ? extractMeaningfulError(fullError) : prev[modelId]?.error,
              fullError: fullError || prev[modelId]?.fullError,
              successResponse: checkData.status === "healthy" ? checkData : prev[modelId]?.successResponse,
            },
          }));
        }
      } catch (dbError) {
        console.debug("Could not fetch updated status from database (non-critical):", dbError);
      }
    } catch (error) {
      const currentTime = new Date().toLocaleString();
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
  };

  const runAllHealthChecks = async () => {
    const modelsToCheck = selectedModelsForHealth.length > 0 ? selectedModelsForHealth : all_models_on_proxy;

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

    const healthCheckResults: { [key: string]: any } = {};

    const healthCheckPromises = modelsToCheck.map(async (modelId) => {
      if (!accessToken) return;

      try {
        const response = await individualModelHealthCheckCall(accessToken, modelId);
        healthCheckResults[modelId] = response;

        const currentTime = new Date().toLocaleString();
        if (response.unhealthy_count > 0 && response.unhealthy_endpoints && response.unhealthy_endpoints.length > 0) {
          const rawError = response.unhealthy_endpoints[0]?.error || "Health check failed";
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
      } catch (error) {
        console.error(`Health check failed for model id ${modelId}:`, error);
        const currentTime = new Date().toLocaleString();
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
    });

    await Promise.allSettled(healthCheckPromises);

    try {
      if (!accessToken) return;
      const latestHealthChecks = await latestHealthChecksCall(accessToken);

      if (latestHealthChecks.latest_health_checks) {
        Object.entries(latestHealthChecks.latest_health_checks).forEach(([modelId, checkData]: [string, any]) => {
          if (modelsToCheck.includes(modelId) && checkData) {
            const fullError = checkData.error_message || undefined;
            setModelHealthStatuses((prev) => {
              const currentStatus = prev[modelId];
              return {
                ...prev,
                [modelId]: {
                  status: checkData.status || currentStatus?.status || "unknown",
                  lastCheck: checkData.checked_at
                    ? new Date(checkData.checked_at).toLocaleString()
                    : currentStatus?.lastCheck || "None",
                  lastSuccess:
                    checkData.status === "healthy"
                      ? checkData.checked_at
                        ? new Date(checkData.checked_at).toLocaleString()
                        : currentStatus?.lastSuccess || "None"
                      : currentStatus?.lastSuccess || "None",
                  loading: false,
                  error: fullError ? extractMeaningfulError(fullError) : currentStatus?.error,
                  fullError: fullError || currentStatus?.fullError,
                  successResponse: checkData.status === "healthy" ? checkData : currentStatus?.successResponse,
                },
              };
            });
          }
        });
      }
    } catch (dbError) {
      console.warn("Failed to fetch updated health statuses from database (non-critical):", dbError);
    }
  };

  const handleModelSelection = (modelId: string, checked: boolean) => {
    if (checked) {
      setSelectedModelsForHealth((prev) => [...prev, modelId]);
    } else {
      setSelectedModelsForHealth((prev) => prev.filter((id) => id !== modelId));
      setAllModelsSelected(false);
    }
  };

  const handleSelectAll = (checked: boolean) => {
    setAllModelsSelected(checked);
    if (checked) {
      setSelectedModelsForHealth(all_models_on_proxy);
    } else {
      setSelectedModelsForHealth([]);
    }
  };

  const getStatusBadge = (status: string) => {
    switch (status) {
      case "healthy":
        return <Badge color="emerald">healthy</Badge>;
      case "unhealthy":
        return <Badge color="red">unhealthy</Badge>;
      case "checking":
        return <Badge color="blue">checking</Badge>;
      case "none":
        return <Badge color="gray">none</Badge>;
      default:
        return <Badge color="gray">unknown</Badge>;
    }
  };

  const showErrorModal = (modelName: string, cleanedError: string, fullError: string) => {
    setSelectedErrorDetails({
      modelName,
      cleanedError,
      fullError,
    });
    setErrorModalVisible(true);
  };

  const closeErrorModal = () => {
    setErrorModalVisible(false);
    setSelectedErrorDetails(null);
  };

  const showSuccessModal = (modelName: string, response: any) => {
    setSelectedSuccessDetails({
      modelName,
      response,
    });
    setSuccessModalVisible(true);
  };

  const closeSuccessModal = () => {
    setSuccessModalVisible(false);
    setSelectedSuccessDetails(null);
  };

  return (
    <div>
      <div className="mb-6">
        <div className="flex justify-between items-center">
          <div>
            <Title>Model Health Status</Title>
            <Text className="text-gray-600 mt-1">
              Run health checks on individual models to verify they are working correctly
            </Text>
          </div>
          <div className="flex items-center gap-3">
            {selectedModelsForHealth.length > 0 && (
              <Button size="sm" variant="light" onClick={() => handleSelectAll(false)} className="px-3 py-1 text-sm">
                Clear Selection
              </Button>
            )}
            <Button
              size="sm"
              variant="secondary"
              onClick={runAllHealthChecks}
              disabled={Object.values(modelHealthStatuses).some((status) => status.loading)}
              className="px-3 py-1 text-sm"
            >
              {selectedModelsForHealth.length > 0 && selectedModelsForHealth.length < all_models_on_proxy.length
                ? "Run Selected Checks"
                : "Run All Checks"}
            </Button>
          </div>
        </div>
      </div>

      <div>
        <ModelDataTable
          columns={healthCheckColumns(
            modelHealthStatuses,
            selectedModelsForHealth,
            allModelsSelected,
            handleModelSelection,
            handleSelectAll,
            runIndividualHealthCheck,
            getStatusBadge,
            getDisplayModelName,
            showErrorModal,
            showSuccessModal,
            setSelectedModelId,
            teams,
          )}
          data={modelData.data.map((model: any) => {
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
          })}
          isLoading={false}
        />
      </div>

      {/* Error Modal */}
      <Modal
        title={selectedErrorDetails ? `Health Check Error - ${selectedErrorDetails.modelName}` : "Error Details"}
        open={errorModalVisible}
        onCancel={closeErrorModal}
        footer={[
          <AntdButton key="close" onClick={closeErrorModal}>
            Close
          </AntdButton>,
        ]}
        width={800}
      >
        {selectedErrorDetails && (
          <div className="space-y-4">
            <div>
              <Text className="font-medium">Error:</Text>
              <div className="mt-2 p-3 bg-red-50 border border-red-200 rounded-md">
                <Text className="text-red-800">{selectedErrorDetails.cleanedError}</Text>
              </div>
            </div>

            <div>
              <Text className="font-medium">Full Error Details:</Text>
              <div className="mt-2 p-3 bg-gray-50 border border-gray-200 rounded-md max-h-96 overflow-y-auto">
                <pre className="text-sm text-gray-800 whitespace-pre-wrap">{selectedErrorDetails.fullError}</pre>
              </div>
            </div>
          </div>
        )}
      </Modal>

      {/* Success Modal */}
      <Modal
        title={
          selectedSuccessDetails ? `Health Check Response - ${selectedSuccessDetails.modelName}` : "Response Details"
        }
        open={successModalVisible}
        onCancel={closeSuccessModal}
        footer={[
          <AntdButton key="close" onClick={closeSuccessModal}>
            Close
          </AntdButton>,
        ]}
        width={800}
      >
        {selectedSuccessDetails && (
          <div className="space-y-4">
            <div>
              <Text className="font-medium">Status:</Text>
              <div className="mt-2 p-3 bg-green-50 border border-green-200 rounded-md">
                <Text className="text-green-800">Health check passed successfully</Text>
              </div>
            </div>

            <div>
              <Text className="font-medium">Response Details:</Text>
              <div className="mt-2 p-3 bg-gray-50 border border-gray-200 rounded-md max-h-96 overflow-y-auto">
                <pre className="text-sm text-gray-800 whitespace-pre-wrap">
                  {JSON.stringify(selectedSuccessDetails.response, null, 2)}
                </pre>
              </div>
            </div>
          </div>
        )}
      </Modal>
    </div>
  );
};

export default HealthCheckComponent;
