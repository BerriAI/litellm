import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
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
  setGlobalRetryPolicy: React.Dispatch<
    React.SetStateAction<GlobalRetryPolicyObject | null>
  >;
  defaultRetry: number;
  modelGroupRetryPolicy: RetryPolicyObject | null;
  setModelGroupRetryPolicy: React.Dispatch<
    React.SetStateAction<RetryPolicyObject | null>
  >;
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
  return (
    <>
      <div className="flex items-center gap-4 mb-6">
        <div className="flex items-center">
          <span>Retry Policy Scope:</span>
          <Select
            value={
              selectedModelGroup === "global"
                ? "global"
                : selectedModelGroup || availableModelGroups[0]
            }
            onValueChange={(value) => setSelectedModelGroup(value)}
          >
            <SelectTrigger className="ml-2 w-48">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="global">Global Default</SelectItem>
              {availableModelGroups.map((group) => (
                <SelectItem key={group} value={group}>
                  {group}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>

      {selectedModelGroup === "global" ? (
        <>
          <h3 className="text-lg font-semibold">Global Retry Policy</h3>
          <p className="mb-6 text-muted-foreground">
            Default retry settings applied to all model groups unless
            overridden
          </p>
        </>
      ) : (
        <>
          <h3 className="text-lg font-semibold">
            Retry Policy for {selectedModelGroup}
          </h3>
          <p className="mb-6 text-muted-foreground">
            Model-specific retry settings. Falls back to global defaults if not
            set.
          </p>
        </>
      )}
      <table>
        <tbody>
          {Object.entries(retryPolicyMap).map(
            ([exceptionType, retryPolicyKey], idx) => {
              let retryCount: number;

              if (selectedModelGroup === "global") {
                retryCount = globalRetryPolicy?.[retryPolicyKey] ?? defaultRetry;
              } else {
                const modelSpecificCount =
                  modelGroupRetryPolicy?.[selectedModelGroup!]?.[
                    retryPolicyKey
                  ];
                if (modelSpecificCount != null) {
                  retryCount = modelSpecificCount;
                } else {
                  retryCount =
                    globalRetryPolicy?.[retryPolicyKey] ?? defaultRetry;
                }
              }

              return (
                <tr
                  key={idx}
                  className="flex justify-between items-center mt-2"
                >
                  <td>
                    <span>{exceptionType}</span>
                    {selectedModelGroup !== "global" && (
                      <span className="text-xs text-muted-foreground ml-2">
                        (Global:{" "}
                        {globalRetryPolicy?.[retryPolicyKey] ?? defaultRetry})
                      </span>
                    )}
                  </td>
                  <td>
                    <Input
                      type="number"
                      className="ml-5 w-24"
                      value={retryCount}
                      min={0}
                      step={1}
                      onChange={(e) => {
                        const raw = e.target.value;
                        const value = raw === "" ? null : Number(raw);
                        if (selectedModelGroup === "global") {
                          setGlobalRetryPolicy((prevGlobalRetryPolicy) => {
                            if (value == null) return prevGlobalRetryPolicy;
                            return {
                              ...(prevGlobalRetryPolicy ?? {}),
                              [retryPolicyKey]: value,
                            };
                          });
                        } else {
                          setModelGroupRetryPolicy(
                            (prevModelGroupRetryPolicy) => {
                              const prevRetryPolicy =
                                prevModelGroupRetryPolicy?.[
                                  selectedModelGroup!
                                ] ?? {};
                              return {
                                ...(prevModelGroupRetryPolicy ?? {}),
                                [selectedModelGroup!]: {
                                  ...prevRetryPolicy,
                                  [retryPolicyKey!]: value,
                                },
                              } as RetryPolicyObject;
                            },
                          );
                        }
                      }}
                    />
                  </td>
                </tr>
              );
            },
          )}
        </tbody>
      </table>
      <Button className="mt-6 mr-8" onClick={handleSaveRetrySettings}>
        Save
      </Button>
    </>
  );
};

export default ModelRetrySettingsTab;
