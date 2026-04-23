import React, { useState, useEffect } from "react";
import { Card } from "@/components/ui/card";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { Form, Select } from "antd";
import { Info } from "lucide-react";
import GuardrailSelector from "../guardrails/GuardrailSelector";

interface PassThroughGuardrailsSectionProps {
  accessToken: string;
  value?: Record<string, { request_fields?: string[]; response_fields?: string[] } | null>;
  onChange?: (guardrails: Record<string, { request_fields?: string[]; response_fields?: string[] } | null>) => void;
  disabled?: boolean;
}

const PassThroughGuardrailsSection: React.FC<PassThroughGuardrailsSectionProps> = ({
  accessToken,
  value = {},
  onChange,
  disabled = false,
}) => {
  const [selectedGuardrails, setSelectedGuardrails] = useState<string[]>(Object.keys(value));
  const [guardrailSettings, setGuardrailSettings] = useState<
    Record<string, { request_fields?: string[]; response_fields?: string[] } | null>
  >(value);

  // Sync external value changes
  useEffect(() => {
    setGuardrailSettings(value);
    setSelectedGuardrails(Object.keys(value));
  }, [value]);

  const handleGuardrailChange = (guardrails: string[]) => {
    setSelectedGuardrails(guardrails);

    // Create new settings object with selected guardrails
    const newSettings: Record<string, { request_fields?: string[]; response_fields?: string[] } | null> = {};
    guardrails.forEach((name) => {
      // Preserve existing settings or set to null (uses entire payload)
      newSettings[name] = guardrailSettings[name] || null;
    });

    setGuardrailSettings(newSettings);
    if (onChange) {
      onChange(newSettings);
    }
  };

  const handleFieldChange = (
    guardrailName: string,
    fieldType: "request_fields" | "response_fields",
    fields: string[]
  ) => {
    const currentSettings = guardrailSettings[guardrailName] || {};
    const newSettings = {
      ...guardrailSettings,
      [guardrailName]: {
        ...currentSettings,
        [fieldType]: fields.length > 0 ? fields : undefined,
      },
    };

    // If no fields are set, set to null (entire payload)
    if (!newSettings[guardrailName]?.request_fields && !newSettings[guardrailName]?.response_fields) {
      newSettings[guardrailName] = null;
    }

    setGuardrailSettings(newSettings);
    if (onChange) {
      onChange(newSettings);
    }
  };

  return (
    <Card className="p-6">
      <h3 className="text-lg font-semibold text-foreground mb-2">Guardrails</h3>
      <p className="text-muted-foreground mb-6">
        Configure guardrails to enforce policies on requests and responses.
        Guardrails are opt-in for passthrough endpoints.
      </p>

      <div className="flex gap-2 items-start p-3 rounded-md bg-blue-50 dark:bg-blue-950/30 border border-blue-200 dark:border-blue-900 text-blue-800 dark:text-blue-200 mb-4">
        <Info className="h-4 w-4 mt-0.5 shrink-0" />
        <div className="flex-1">
          <div className="font-semibold">
            Field-Level Targeting{" "}
            <a
              href="https://docs.litellm.ai/docs/proxy/pass_through_guardrails#field-level-targeting"
              target="_blank"
              rel="noopener noreferrer"
              className="underline"
            >
              (Learn More)
            </a>
          </div>
          <div className="space-y-2 mt-2">
            <div className="text-sm">
              Optionally specify which fields to check. If left empty, the
              entire request/response is sent to the guardrail.
            </div>
            <div className="text-xs space-y-1 mt-2">
              <div className="font-medium">Common Examples:</div>
              <div>
                •{" "}
                <code className="bg-background px-1 rounded">query</code> -
                Single field
              </div>
              <div>
                •{" "}
                <code className="bg-background px-1 rounded">
                  documents[*].text
                </code>{" "}
                - All text in documents array
              </div>
              <div>
                •{" "}
                <code className="bg-background px-1 rounded">
                  messages[*].content
                </code>{" "}
                - All message contents
              </div>
            </div>
          </div>
        </div>
      </div>

      <Form.Item
        label={
          <span className="text-sm font-medium text-foreground flex items-center">
            Select Guardrails
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Info className="ml-2 h-3 w-3 inline text-primary cursor-help" />
                </TooltipTrigger>
                <TooltipContent className="max-w-xs">
                  Choose which guardrails should run on this endpoint. Org/team/
                  key level guardrails will also be included.
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          </span>
        }
      >
        <GuardrailSelector
          accessToken={accessToken}
          value={selectedGuardrails}
          onChange={handleGuardrailChange}
          disabled={disabled}
        />
      </Form.Item>

      {selectedGuardrails.length > 0 && (
        <div className="mt-6 space-y-4">
          <div className="flex items-center justify-between mb-3">
            <div className="text-sm font-medium text-foreground">
              Field Targeting (Optional)
            </div>
            <div className="text-xs text-muted-foreground">
              💡 Tip: Leave empty to check entire payload
            </div>
          </div>
          {selectedGuardrails.map((guardrailName) => (
            <Card key={guardrailName} className="p-4 bg-muted">
              <div className="text-sm font-medium text-foreground mb-3">
                {guardrailName}
              </div>
              <div className="space-y-3">
                <div>
                  <div className="flex items-center justify-between mb-1">
                    <label className="text-xs text-muted-foreground flex items-center">
                      Request Fields (pre_call)
                      <TooltipProvider>
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <Info className="ml-1 h-3 w-3 inline text-muted-foreground" />
                          </TooltipTrigger>
                          <TooltipContent>
                            <div>
                              <div className="font-medium mb-1">
                                Specify which request fields to check
                              </div>
                              <div className="text-xs space-y-1">
                                <div>Examples:</div>
                                <div>• query</div>
                                <div>• documents[*].text</div>
                                <div>• messages[*].content</div>
                              </div>
                            </div>
                          </TooltipContent>
                        </Tooltip>
                      </TooltipProvider>
                    </label>
                    <div className="flex gap-1">
                      <button
                        type="button"
                        onClick={() => {
                          const current =
                            guardrailSettings[guardrailName]?.request_fields ||
                            [];
                          handleFieldChange(guardrailName, "request_fields", [
                            ...current,
                            "query",
                          ]);
                        }}
                        className="text-xs px-2 py-1 bg-background border border-border rounded hover:bg-muted"
                        disabled={disabled}
                      >
                        + query
                      </button>
                      <button
                        type="button"
                        onClick={() => {
                          const current =
                            guardrailSettings[guardrailName]?.request_fields ||
                            [];
                          handleFieldChange(guardrailName, "request_fields", [
                            ...current,
                            "documents[*]",
                          ]);
                        }}
                        className="text-xs px-2 py-1 bg-background border border-border rounded hover:bg-muted"
                        disabled={disabled}
                      >
                        + documents[*]
                      </button>
                    </div>
                  </div>
                  <Select
                    mode="tags"
                    style={{ width: "100%" }}
                    placeholder="Type field name or use + buttons above (e.g., query, documents[*].text)"
                    value={
                      guardrailSettings[guardrailName]?.request_fields || []
                    }
                    onChange={(fields) =>
                      handleFieldChange(guardrailName, "request_fields", fields)
                    }
                    disabled={disabled}
                    tokenSeparators={[","]}
                  />
                </div>
                <div>
                  <div className="flex items-center justify-between mb-1">
                    <label className="text-xs text-muted-foreground flex items-center">
                      Response Fields (post_call)
                      <TooltipProvider>
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <Info className="ml-1 h-3 w-3 inline text-muted-foreground" />
                          </TooltipTrigger>
                          <TooltipContent>
                            <div>
                              <div className="font-medium mb-1">
                                Specify which response fields to check
                              </div>
                              <div className="text-xs space-y-1">
                                <div>Examples:</div>
                                <div>• results[*].text</div>
                                <div>• choices[*].message.content</div>
                              </div>
                            </div>
                          </TooltipContent>
                        </Tooltip>
                      </TooltipProvider>
                    </label>
                    <div className="flex gap-1">
                      <button
                        type="button"
                        onClick={() => {
                          const current =
                            guardrailSettings[guardrailName]?.response_fields ||
                            [];
                          handleFieldChange(guardrailName, "response_fields", [
                            ...current,
                            "results[*]",
                          ]);
                        }}
                        className="text-xs px-2 py-1 bg-background border border-border rounded hover:bg-muted"
                        disabled={disabled}
                      >
                        + results[*]
                      </button>
                    </div>
                  </div>
                  <Select
                    mode="tags"
                    style={{ width: "100%" }}
                    placeholder="Type field name or use + buttons above (e.g., results[*].text)"
                    value={
                      guardrailSettings[guardrailName]?.response_fields || []
                    }
                    onChange={(fields) =>
                      handleFieldChange(
                        guardrailName,
                        "response_fields",
                        fields,
                      )
                    }
                    disabled={disabled}
                    tokenSeparators={[","]}
                  />
                </div>
              </div>
            </Card>
          ))}
        </div>
      )}
    </Card>
  );
};

export default PassThroughGuardrailsSection;

