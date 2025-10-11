import { Button, Select, SelectItem, TabPanel, Text, Title } from "@tremor/react";
import { InputNumber } from "antd";
import React from "react";

interface GlobalRetryPolicyObject {
  [retryPolicyKey: string]: number;
}

interface RetryPolicyObject {
  [key: string]: { [retryPolicyKey: string]: number } | undefined;
}

interface ModelRetrySettingsTabProps {
  selectedModelGroup: string | null;
  setSelectedModelGroup: (selectedModelGroup: string | null) => void;
  availableModelGroups: string[];
  globalRetryPolicy: GlobalRetryPolicyObject | null;
  setGlobalRetryPolicy: React.Dispatch<React.SetStateAction<GlobalRetryPolicyObject | null>>;
  defaultRetry: number;
  modelGroupRetryPolicy: RetryPolicyObject | null;
  setModelGroupRetryPolicy: React.Dispatch<React.SetStateAction<RetryPolicyObject | null>>;
  handleSaveRetrySettings: () => void;
}

const retryPolicyMap: Record<string, string> = {
  "BadRequestError (400)": "BadRequestErrorRetries",
  "AuthenticationError  (401)": "AuthenticationErrorRetries",
  "TimeoutError (408)": "TimeoutErrorRetries",
  "RateLimitError (429)": "RateLimitErrorRetries",
  "ContentPolicyViolationError (400)": "ContentPolicyViolationErrorRetries",
  "InternalServerError (500)": "InternalServerErrorRetries",
};

const ModelRetrySettingsTab = ({
  selectedModelGroup,
  setSelectedModelGroup,
  availableModelGroups,
  globalRetryPolicy,
  setGlobalRetryPolicy,
  defaultRetry,
  modelGroupRetryPolicy,
  setModelGroupRetryPolicy,
  handleSaveRetrySettings,
}: ModelRetrySettingsTabProps) => {
  //  const [modelGroupRetryPolicy, setModelGroupRetryPolicy] = useState<RetryPolicyObject | null>(null);

  return (
    <TabPanel>
      <div className="flex items-center gap-4 mb-6">
        <div className="flex items-center">
          <Text>Retry Policy Scope:</Text>
          <Select
            className="ml-2 w-48"
            defaultValue="global"
            value={selectedModelGroup === "global" ? "global" : selectedModelGroup || availableModelGroups[0]}
            onValueChange={(value) => setSelectedModelGroup(value)}
          >
            <SelectItem value="global">Global Default</SelectItem>
            {availableModelGroups.map((group, idx) => (
              <SelectItem key={idx} value={group} onClick={() => setSelectedModelGroup(group)}>
                {group}
              </SelectItem>
            ))}
          </Select>
        </div>
      </div>

      {selectedModelGroup === "global" ? (
        <>
          <Title>Global Retry Policy</Title>
          <Text className="mb-6">Default retry settings applied to all model groups unless overridden</Text>
        </>
      ) : (
        <>
          <Title>Retry Policy for {selectedModelGroup}</Title>
          <Text className="mb-6">Model-specific retry settings. Falls back to global defaults if not set.</Text>
        </>
      )}
      {retryPolicyMap && (
        <table>
          <tbody>
            {Object.entries(retryPolicyMap).map(([exceptionType, retryPolicyKey], idx) => {
              let retryCount: number;

              if (selectedModelGroup === "global") {
                // Show global policy values
                retryCount = globalRetryPolicy?.[retryPolicyKey] ?? defaultRetry;
              } else {
                // Show model-group specific values with fallback to global
                const modelSpecificCount = modelGroupRetryPolicy?.[selectedModelGroup!]?.[retryPolicyKey];
                if (modelSpecificCount != null) {
                  retryCount = modelSpecificCount;
                } else {
                  // Fall back to global policy, then default
                  retryCount = globalRetryPolicy?.[retryPolicyKey] ?? defaultRetry;
                }
              }

              return (
                <tr key={idx} className="flex justify-between items-center mt-2">
                  <td>
                    <Text>{exceptionType}</Text>
                    {selectedModelGroup !== "global" && (
                      <Text className="text-xs text-gray-500 ml-2">
                        (Global: {globalRetryPolicy?.[retryPolicyKey] ?? defaultRetry})
                      </Text>
                    )}
                  </td>
                  <td>
                    <InputNumber
                      className="ml-5"
                      value={retryCount}
                      min={0}
                      step={1}
                      onChange={(value) => {
                        if (selectedModelGroup === "global") {
                          // Update global policy
                          setGlobalRetryPolicy((prevGlobalRetryPolicy) => {
                            if (value == null) return prevGlobalRetryPolicy;
                            return {
                              ...(prevGlobalRetryPolicy ?? {}),
                              [retryPolicyKey]: value,
                            };
                          });
                        } else {
                          // Update model-group specific policy
                          setModelGroupRetryPolicy((prevModelGroupRetryPolicy) => {
                            const prevRetryPolicy = prevModelGroupRetryPolicy?.[selectedModelGroup!] ?? {};
                            return {
                              ...(prevModelGroupRetryPolicy ?? {}),
                              [selectedModelGroup!]: {
                                ...prevRetryPolicy,
                                [retryPolicyKey!]: value,
                              },
                            } as RetryPolicyObject;
                          });
                        }
                      }}
                    />
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
      <Button className="mt-6 mr-8" onClick={handleSaveRetrySettings}>
        Save
      </Button>
    </TabPanel>
  );
};

export default ModelRetrySettingsTab;
