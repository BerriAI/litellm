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

  // Initialize health statuses on component mount
  useEffect(() => {
    if (!accessToken || !modelData?.data) return;

    const initializeHealthStatuses = async () => {
      const healthStatusMap: { [key: string]: HealthStatus } = {};

      // Initialize all models with default state using model names
      modelData.data.forEach((model: any) => {
        const modelName = model.model_name;
        healthStatusMap[modelName] = {
          status: "none",
          lastCheck: "None",
          lastSuccess: "None",
          loading: false,
          error: undefined,
          fullError: undefined,
          successResponse: undefined,
        };
      });

      try {
        const latestHealthChecks = await latestHealthChecksCall(accessToken);

        // Override with actual database data if it exists
        if (
          latestHealthChecks &&
          latestHealthChecks.latest_health_checks &&
          typeof latestHealthChecks.latest_health_checks === "object"
        ) {
          Object.entries(latestHealthChecks.latest_health_checks).forEach(([key, checkData]: [string, any]) => {
            if (!checkData) return;

            let targetModelName: string | null = null;

            // The key could be either model_id or model_name, try both approaches
            const directModelMatch = modelData.data.find((m: any) => m.model_name === key);
            if (directModelMatch) {
              targetModelName = directModelMatch.model_name;
            } else {
              // If not a direct match, treat as model_id and find the corresponding model
              const modelByIdMatch = modelData.data.find((m: any) => m.model_info && m.model_info.id === key);
              if (modelByIdMatch) {
                targetModelName = modelByIdMatch.model_name;
              } else {
                // Check if checkData contains model_name and use that
                if (checkData.model_name) {
                  const modelByNameInData = modelData.data.find((m: any) => m.model_name === checkData.model_name);
                  if (modelByNameInData) {
                    targetModelName = modelByNameInData.model_name;
                  }
                }
              }
            }

            if (targetModelName) {
              const fullError = checkData.error_message || undefined;

              healthStatusMap[targetModelName] = {
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
            }
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

  const runIndividualHealthCheck = async (modelName: string) => {
    if (!accessToken) return;

    setModelHealthStatuses((prev) => ({
      ...prev,
      [modelName]: {
        ...prev[modelName],
        loading: true,
        status: "checking",
      },
    }));

    try {
      // Run the health check and process the response directly
      const response = await individualModelHealthCheckCall(accessToken, modelName);
      const currentTime = new Date().toLocaleString();

      // Check if there are any unhealthy endpoints (which means this specific model failed)
      if (response.unhealthy_count > 0 && response.unhealthy_endpoints && response.unhealthy_endpoints.length > 0) {
        const rawError = response.unhealthy_endpoints[0]?.error || "Health check failed";
        const errorMessage = extractMeaningfulError(rawError);
        setModelHealthStatuses((prev) => ({
          ...prev,
          [modelName]: {
            status: "unhealthy",
            lastCheck: currentTime,
            lastSuccess: prev[modelName]?.lastSuccess || "None",
            loading: false,
            error: errorMessage,
            fullError: rawError,
          },
        }));
      } else {
        setModelHealthStatuses((prev) => ({
          ...prev,
          [modelName]: {
            status: "healthy",
            lastCheck: currentTime,
            lastSuccess: currentTime,
            loading: false,
            successResponse: response,
          },
        }));
      }

      // Refresh health status from database to get the saved check data including timestamp
      try {
        const latestHealthChecks = await latestHealthChecksCall(accessToken);

        // Find the model ID for this model name to look up database data
        const model = modelData.data.find((m: any) => m.model_name === modelName);
        if (model) {
          const modelId = model.model_info.id;
          const checkData = latestHealthChecks.latest_health_checks?.[modelId];

          if (checkData) {
            const fullError = checkData.error_message || undefined;
            setModelHealthStatuses((prev) => ({
              ...prev,
              [modelName]: {
                status: checkData.status || prev[modelName]?.status || "unknown",
                lastCheck: checkData.checked_at
                  ? new Date(checkData.checked_at).toLocaleString()
                  : prev[modelName]?.lastCheck || "None",
                lastSuccess:
                  checkData.status === "healthy"
                    ? checkData.checked_at
                      ? new Date(checkData.checked_at).toLocaleString()
                      : prev[modelName]?.lastSuccess || "None"
                    : prev[modelName]?.lastSuccess || "None",
                loading: false,
                error: fullError ? extractMeaningfulError(fullError) : prev[modelName]?.error,
                fullError: fullError || prev[modelName]?.fullError,
                successResponse: checkData.status === "healthy" ? checkData : prev[modelName]?.successResponse,
              },
            }));
          }
        }
      } catch (dbError) {
        // Ignore database errors - we already have the health check result from the API call
        console.debug("Could not fetch updated status from database (non-critical):", dbError);
      }
    } catch (error) {
      const currentTime = new Date().toLocaleString();
      const rawError = error instanceof Error ? error.message : String(error);
      const errorMessage = extractMeaningfulError(rawError);
      setModelHealthStatuses((prev) => ({
        ...prev,
        [modelName]: {
          status: "unhealthy",
          lastCheck: currentTime,
          lastSuccess: prev[modelName]?.lastSuccess || "None",
          loading: false,
          error: errorMessage,
          fullError: rawError,
        },
      }));
    }
  };

  const runAllHealthChecks = async () => {
    const modelsToCheck = selectedModelsForHealth.length > 0 ? selectedModelsForHealth : all_models_on_proxy;

    // Set all models to loading state
    const loadingStatuses = modelsToCheck.reduce(
      (acc, modelName) => {
        acc[modelName] = {
          ...modelHealthStatuses[modelName],
          loading: true,
          status: "checking",
        };
        return acc;
      },
      {} as typeof modelHealthStatuses,
    );

    setModelHealthStatuses((prev) => ({ ...prev, ...loadingStatuses }));

    // Store results from individual health checks
    const healthCheckResults: { [key: string]: any } = {};

    // Run all health checks in parallel and collect results
    const healthCheckPromises = modelsToCheck.map(async (modelName) => {
      if (!accessToken) return;

      try {
        // Run the health check and store the result
        const response = await individualModelHealthCheckCall(accessToken, modelName);
        healthCheckResults[modelName] = response;

        // Update status immediately based on response
        const currentTime = new Date().toLocaleString();
        // Check if there are any unhealthy endpoints (which means this specific model failed)
        if (response.unhealthy_count > 0 && response.unhealthy_endpoints && response.unhealthy_endpoints.length > 0) {
          const rawError = response.unhealthy_endpoints[0]?.error || "Health check failed";
          const errorMessage = extractMeaningfulError(rawError);
          setModelHealthStatuses((prev) => ({
            ...prev,
            [modelName]: {
              status: "unhealthy",
              lastCheck: currentTime,
              lastSuccess: prev[modelName]?.lastSuccess || "None",
              loading: false,
              error: errorMessage,
              fullError: rawError,
            },
          }));
        } else {
          setModelHealthStatuses((prev) => ({
            ...prev,
            [modelName]: {
              status: "healthy",
              lastCheck: currentTime,
              lastSuccess: currentTime,
              loading: false,
              successResponse: response,
            },
          }));
        }
      } catch (error) {
        console.error(`Health check failed for ${modelName}:`, error);
        // Set error status for failed health checks
        const currentTime = new Date().toLocaleString();
        const rawError = error instanceof Error ? error.message : String(error);
        const errorMessage = extractMeaningfulError(rawError);
        setModelHealthStatuses((prev) => ({
          ...prev,
          [modelName]: {
            status: "unhealthy",
            lastCheck: currentTime,
            lastSuccess: prev[modelName]?.lastSuccess || "None",
            loading: false,
            error: errorMessage,
            fullError: rawError,
          },
        }));
      }
    });

    // Wait for all health checks to complete
    await Promise.allSettled(healthCheckPromises);

    // Refresh health statuses from database to get the saved check data including timestamps
    try {
      if (!accessToken) return;
      const latestHealthChecks = await latestHealthChecksCall(accessToken);

      if (latestHealthChecks.latest_health_checks) {
        // Update health statuses from database, which should have the most accurate saved data
        Object.entries(latestHealthChecks.latest_health_checks).forEach(([modelId, checkData]: [string, any]) => {
          // Find the model name for this model ID
          const model = modelData.data.find((m: any) => m.model_info.id === modelId);
          if (model && modelsToCheck.includes(model.model_name) && checkData) {
            const modelName = model.model_name;
            const fullError = checkData.error_message || undefined;
            setModelHealthStatuses((prev) => {
              const currentStatus = prev[modelName];
              return {
                ...prev,
                [modelName]: {
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
      // This is non-critical - we already have the health check results from the API calls
    }
  };

  const handleModelSelection = (modelName: string, checked: boolean) => {
    if (checked) {
      setSelectedModelsForHealth((prev) => [...prev, modelName]);
    } else {
      setSelectedModelsForHealth((prev) => prev.filter((name) => name !== modelName));
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
            const modelName = model.model_name;
            const healthStatus = modelHealthStatuses[modelName] || {
              status: "none",
              lastCheck: "None",
              loading: false,
            };
            return {
              model_name: model.model_name,
              model_info: model.model_info,
              provider: model.provider,
              litellm_model_name: model.litellm_model_name,
              health_status: healthStatus.status,
              last_check: healthStatus.lastCheck,
              last_success: healthStatus.lastSuccess || "None",
              health_loading: healthStatus.loading,
              health_error: healthStatus.error,
              health_full_error: healthStatus.fullError,
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
