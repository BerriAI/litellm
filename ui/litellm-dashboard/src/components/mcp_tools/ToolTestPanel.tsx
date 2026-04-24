import React from "react";
import { useForm, Controller } from "react-hook-form";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { Info as InfoCircleOutlined, X, Copy, Check, AlertCircle, Zap } from "lucide-react";
import { MCPTool, InputSchema, InputSchemaProperty } from "./types";
import NotificationsManager from "../molecules/notifications_manager";

const isPlainObject = (value: unknown): value is Record<string, any> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

function buildArrayItems(items?: InputSchemaProperty | InputSchemaProperty[]): any[] {
  if (!items) {
    return [];
  }

  if (Array.isArray(items)) {
    return items
      .map((item) => buildDefaultValue(item))
      .filter((value) => value !== undefined);
  }

  const itemDefault = buildDefaultValue(items);
  if (itemDefault === undefined) {
    return [];
  }

  return [itemDefault];
}

function buildDefaultValue(prop?: InputSchemaProperty, overrideDefault?: any): any {
  if (!prop) {
    return undefined;
  }

  const effectiveDefault = overrideDefault !== undefined ? overrideDefault : prop.default;

  if (prop.type === "object") {
    const base = isPlainObject(effectiveDefault) ? { ...effectiveDefault } : {};

    if (prop.properties) {
      Object.entries(prop.properties).forEach(([childKey, childProp]) => {
        base[childKey] = buildDefaultValue(childProp, base[childKey]);
      });
    }

    return base;
  }

  if (prop.type === "array") {
    if (Array.isArray(effectiveDefault)) {
      const itemSchema = prop.items;
      if (!itemSchema) {
        return effectiveDefault;
      }

      if (effectiveDefault.length === 0) {
        const sample = buildArrayItems(itemSchema);
        return sample.length ? sample : effectiveDefault;
      }

      if (Array.isArray(itemSchema)) {
        return effectiveDefault.map((value, index) => {
          const schema = itemSchema[index] ?? itemSchema[itemSchema.length - 1];
          return buildDefaultValue(schema, value);
        });
      }

      return effectiveDefault.map((value) => buildDefaultValue(itemSchema, value));
    }

    if (effectiveDefault !== undefined) {
      return effectiveDefault;
    }

    return buildArrayItems(prop.items);
  }

  if (effectiveDefault !== undefined) {
    return effectiveDefault;
  }

  switch (prop.type) {
    case "integer":
    case "number":
      return 0;
    case "boolean":
      return false;
    case "string":
    default:
      return "";
  }
}

const getInitialValueForField = (prop: InputSchemaProperty): any => {
  const defaultValue = buildDefaultValue(prop);
  if (prop.type === "object" || prop.type === "array") {
    const fallback = prop.type === "array" ? [] : {};
    return JSON.stringify(defaultValue ?? fallback, null, 2);
  }
  return defaultValue;
};

function validateField(
  key: string,
  prop: InputSchemaProperty,
  required: boolean,
  value: any,
): string | true {
  if (required && (value === undefined || value === null || value === "")) {
    return `Please enter ${key}`;
  }
  if (prop.type === "object" || prop.type === "array") {
    if ((value === undefined || value === null || value === "") && !required) {
      return true;
    }
    try {
      const parsed = typeof value === "string" ? JSON.parse(value) : value;
      const isValidObject =
        prop.type === "object" &&
        parsed !== null &&
        typeof parsed === "object" &&
        !Array.isArray(parsed);
      const isValidArray = prop.type === "array" && Array.isArray(parsed);
      if ((prop.type === "object" && isValidObject) || (prop.type === "array" && isValidArray)) {
        return true;
      }
      return prop.type === "object" ? "Please enter a JSON object" : "Please enter a JSON array";
    } catch {
      return "Invalid JSON";
    }
  }
  return true;
}

export function ToolTestPanel({
  tool,
  onSubmit,
  isLoading,
  result,
  error,
  onClose,
}: {
  tool: MCPTool;
  onSubmit: (args: Record<string, any>) => void;
  isLoading: boolean;
  result: any | null;
  error: Error | null;
  onClose: () => void;
}) {
  const [viewMode, setViewMode] = React.useState<"formatted" | "json">("formatted");
  const [startTime, setStartTime] = React.useState<number | null>(null);
  const [duration, setDuration] = React.useState<number | null>(null);

  const schema: InputSchema = React.useMemo(() => {
    if (typeof tool.inputSchema === "string") {
      return {
        type: "object",
        properties: {
          input: {
            type: "string",
            description: "Input for this tool",
          },
        },
        required: ["input"],
      };
    }
    return tool.inputSchema as InputSchema;
  }, [tool.inputSchema]);

  const actualSchema: InputSchema = React.useMemo(() => {
    if (
      schema.properties &&
      schema.properties.params &&
      schema.properties.params.type === "object" &&
      schema.properties.params.properties
    ) {
      return {
        type: "object",
        properties: schema.properties.params.properties,
        required: schema.properties.params.required || [],
      };
    }
    return schema;
  }, [schema]);

  const defaultValues = React.useMemo(() => {
    const values: Record<string, any> = {};
    if (actualSchema.properties) {
      Object.entries(actualSchema.properties).forEach(([key, prop]) => {
        values[key] = getInitialValueForField(prop);
      });
    }
    return values;
  }, [actualSchema]);

  const form = useForm<Record<string, any>>({
    defaultValues,
    mode: "onSubmit",
  });
  const { control, handleSubmit, reset } = form;

  React.useEffect(() => {
    reset(defaultValues);
  }, [defaultValues, reset, tool]);

  const handleFormSubmit = (values: Record<string, any>) => {
    const start = Date.now();
    setStartTime(start);
    setDuration(null);

    const convertedValues: Record<string, any> = {};
    const schemaToUse = actualSchema;

    Object.entries(values).forEach(([key, value]) => {
      const prop = schemaToUse.properties?.[key];
      if (prop && value !== null && value !== undefined && value !== "") {
        switch (prop.type) {
          case "boolean":
            convertedValues[key] = value === "true" || value === true;
            break;
          case "number":
          case "integer": {
            const numericValue = Number(value);
            convertedValues[key] = Number.isNaN(numericValue)
              ? value
              : prop.type === "integer"
                ? Math.trunc(numericValue)
                : numericValue;
            break;
          }
          case "object":
          case "array": {
            try {
              const parsed = typeof value === "string" ? JSON.parse(value) : value;
              const isValidObject =
                prop.type === "object" && parsed !== null && typeof parsed === "object" && !Array.isArray(parsed);
              const isValidArray = prop.type === "array" && Array.isArray(parsed);
              if ((prop.type === "object" && isValidObject) || (prop.type === "array" && isValidArray)) {
                convertedValues[key] = parsed;
              } else {
                convertedValues[key] = value;
              }
            } catch (err) {
              convertedValues[key] = value;
            }
            break;
          }
          case "string":
            convertedValues[key] = String(value);
            break;
          default:
            convertedValues[key] = value;
        }
      } else if (value !== null && value !== undefined && value !== "") {
        convertedValues[key] = value;
      }
    });

    const submitValues =
      schema.properties &&
      schema.properties.params &&
      schema.properties.params.type === "object" &&
      schema.properties.params.properties
        ? { params: convertedValues }
        : convertedValues;

    onSubmit(submitValues);
  };

  React.useEffect(() => {
    if (startTime && (result || error)) {
      const endTime = Date.now();
      setDuration(endTime - startTime);
    }
  }, [result, error, startTime]);

  const copyToClipboard = async (text: string) => {
    try {
      if (navigator.clipboard && window.isSecureContext) {
        await navigator.clipboard.writeText(text);
        return true;
      } else {
        const textArea = document.createElement("textarea");
        textArea.value = text;
        textArea.style.position = "fixed";
        textArea.style.opacity = "0";
        document.body.appendChild(textArea);
        textArea.focus();
        textArea.select();

        const successful = document.execCommand("copy");
        document.body.removeChild(textArea);

        if (!successful) {
          throw new Error("execCommand failed");
        }
        return true;
      }
    } catch (error) {
      console.error("Copy failed:", error);
      return false;
    }
  };

  const handleCopyResult = async () => {
    const success = await copyToClipboard(JSON.stringify(result, null, 2));
    if (success) {
      NotificationsManager.success("Result copied to clipboard");
    } else {
      NotificationsManager.fromBackend("Failed to copy result");
    }
  };

  const handleCopyToolName = async () => {
    const success = await copyToClipboard(tool.name);
    if (success) {
      NotificationsManager.success("Tool name copied to clipboard");
    } else {
      NotificationsManager.fromBackend("Failed to copy tool name");
    }
  };

  return (
    <div className="space-y-4 h-full">
      {/* Compact Header */}
      <div className="flex items-center justify-between pb-3 border-b border-border">
        <div className="flex items-center space-x-3">
          {tool.mcp_info.logo_url && (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={tool.mcp_info.logo_url}
              alt={`${tool.mcp_info.server_name} logo`}
              className="w-6 h-6 object-contain"
            />
          )}
          <div className="flex-1 min-w-0">
            <div className="flex items-center space-x-2 mb-1">
              <h2 className="text-lg font-semibold text-foreground">Test Tool:</h2>
              <div
                className="group inline-flex items-center space-x-1 bg-muted hover:bg-muted/70 px-3 py-1 rounded-md cursor-pointer transition-colors border border-border"
                onClick={handleCopyToolName}
                title="Click to copy tool name"
              >
                <span className="font-mono text-foreground font-medium text-sm">{tool.name}</span>
                <Copy className="w-3 h-3 text-muted-foreground group-hover:text-foreground transition-colors" />
              </div>
            </div>
            <p className="text-xs text-muted-foreground">{tool.description}</p>
            <p className="text-xs text-muted-foreground">Provider: {tool.mcp_info.server_name}</p>
          </div>
        </div>
        <Button
          onClick={onClose}
          variant="ghost"
          size="sm"
          className="text-muted-foreground hover:text-foreground"
        >
          <X className="w-4 h-4" />
        </Button>
      </div>

      {/* Two Column Layout */}
      <div className="grid grid-cols-2 gap-4 h-full">
        {/* Left Column - Input Parameters */}
        <div className="bg-background border border-border rounded-lg">
          <div className="border-b border-border px-4 py-2">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-semibold text-foreground">Input Parameters</h3>
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <InfoCircleOutlined className="text-muted-foreground hover:text-foreground h-4 w-4" />
                  </TooltipTrigger>
                  <TooltipContent>Configure the input parameters for this tool call</TooltipContent>
                </Tooltip>
              </TooltipProvider>
            </div>
          </div>

          <div className="p-4">
            <form onSubmit={handleSubmit(handleFormSubmit)} className="space-y-3">
              {typeof tool.inputSchema === "string" ? (
                <div className="space-y-2">
                  <Label htmlFor={`${tool.name}-input`}>
                    <span className="text-sm font-medium text-foreground">
                      Input <span className="text-destructive">*</span>
                    </span>
                  </Label>
                  <Controller
                    control={control}
                    name="input"
                    rules={{ required: "Please enter input for this tool" }}
                    render={({ field, fieldState }) => (
                      <>
                        <Input
                          id={`${tool.name}-input`}
                          placeholder="Enter input for this tool"
                          {...field}
                          value={field.value ?? ""}
                        />
                        {fieldState.error && (
                          <p className="text-sm text-destructive">{fieldState.error.message}</p>
                        )}
                      </>
                    )}
                  />
                </div>
              ) : actualSchema.properties === undefined ? (
                <div className="text-center py-6 bg-muted rounded-lg border border-border">
                  <div className="max-w-sm mx-auto">
                    <h4 className="text-sm font-medium text-foreground mb-1">No Parameters Required</h4>
                    <p className="text-xs text-muted-foreground">
                      This tool can be called without any input parameters.
                    </p>
                  </div>
                </div>
              ) : (
                <div className="space-y-3">
                  {Object.entries(actualSchema.properties).map(([key, prop]) => {
                    const required = !!actualSchema.required?.includes(key);
                    const fieldKey = `${tool.name}-${key}`;
                    return (
                      <div key={fieldKey} className="space-y-2">
                        <Label htmlFor={fieldKey}>
                          <span className="text-sm font-medium text-foreground flex items-center">
                            {key} {required && <span className="text-destructive">*</span>}
                            {prop.description && (
                              <TooltipProvider>
                                <Tooltip>
                                  <TooltipTrigger asChild>
                                    <InfoCircleOutlined className="ml-2 text-muted-foreground hover:text-foreground h-4 w-4" />
                                  </TooltipTrigger>
                                  <TooltipContent className="max-w-xs">{prop.description}</TooltipContent>
                                </Tooltip>
                              </TooltipProvider>
                            )}
                          </span>
                        </Label>
                        <Controller
                          control={control}
                          name={key}
                          rules={{ validate: (value) => validateField(key, prop, required, value) }}
                          render={({ field, fieldState }) => {
                            const errorMessage = fieldState.error?.message;
                            if (prop.type === "string" && prop.enum) {
                              return (
                                <>
                                  <Select
                                    value={field.value ? String(field.value) : ""}
                                    onValueChange={(v) => field.onChange(v)}
                                  >
                                    <SelectTrigger id={fieldKey} aria-label={key}>
                                      <SelectValue placeholder={`Select ${key}`} />
                                    </SelectTrigger>
                                    <SelectContent>
                                      {prop.enum!.map((v) => (
                                        <SelectItem key={String(v)} value={String(v)}>
                                          {String(v)}
                                        </SelectItem>
                                      ))}
                                    </SelectContent>
                                  </Select>
                                  {errorMessage && (
                                    <p className="text-sm text-destructive">{errorMessage}</p>
                                  )}
                                </>
                              );
                            }
                            if (prop.type === "string") {
                              return (
                                <>
                                  <Input
                                    id={fieldKey}
                                    placeholder={prop.description || `Enter ${key}`}
                                    {...field}
                                    value={field.value ?? ""}
                                  />
                                  {errorMessage && (
                                    <p className="text-sm text-destructive">{errorMessage}</p>
                                  )}
                                </>
                              );
                            }
                            if (prop.type === "number" || prop.type === "integer") {
                              return (
                                <>
                                  <Input
                                    id={fieldKey}
                                    type="number"
                                    step={prop.type === "integer" ? 1 : "any"}
                                    placeholder={prop.description || `Enter ${key}`}
                                    value={field.value ?? ""}
                                    onChange={(e) => {
                                      const v = e.target.value;
                                      field.onChange(v === "" ? "" : Number(v));
                                    }}
                                    onBlur={field.onBlur}
                                  />
                                  {errorMessage && (
                                    <p className="text-sm text-destructive">{errorMessage}</p>
                                  )}
                                </>
                              );
                            }
                            if (prop.type === "boolean") {
                              const valueStr =
                                field.value === true ? "true" : field.value === false ? "false" : "";
                              const valueTitle =
                                field.value === true ? "True" : field.value === false ? "False" : undefined;
                              return (
                                <>
                                  <Select
                                    value={valueStr}
                                    onValueChange={(v) => field.onChange(v === "true")}
                                  >
                                    <SelectTrigger id={fieldKey} aria-label={key} title={valueTitle}>
                                      <SelectValue placeholder={`Select ${key}`} />
                                    </SelectTrigger>
                                    <SelectContent>
                                      <SelectItem value="true">True</SelectItem>
                                      <SelectItem value="false">False</SelectItem>
                                    </SelectContent>
                                  </Select>
                                  {errorMessage && (
                                    <p className="text-sm text-destructive">{errorMessage}</p>
                                  )}
                                </>
                              );
                            }
                            if (prop.type === "object" || prop.type === "array") {
                              return (
                                <div className="space-y-2">
                                  <Textarea
                                    id={fieldKey}
                                    rows={prop.type === "object" ? 6 : 4}
                                    placeholder={
                                      prop.description ||
                                      (prop.type === "object"
                                        ? `Enter JSON object for ${key}`
                                        : `Enter JSON array for ${key}`)
                                    }
                                    spellCheck={false}
                                    data-testid={`textarea-${key}`}
                                    className="font-mono"
                                    {...field}
                                    value={field.value ?? ""}
                                  />
                                  <p className="text-xs text-muted-foreground">
                                    {prop.type === "object"
                                      ? "Provide a valid JSON object."
                                      : "Provide a valid JSON array."}
                                  </p>
                                  {errorMessage && (
                                    <p className="text-sm text-destructive">{errorMessage}</p>
                                  )}
                                </div>
                              );
                            }
                            return (
                              <>
                                <Input
                                  id={fieldKey}
                                  placeholder={prop.description || `Enter ${key}`}
                                  {...field}
                                  value={field.value ?? ""}
                                />
                                {errorMessage && (
                                  <p className="text-sm text-destructive">{errorMessage}</p>
                                )}
                              </>
                            );
                          }}
                        />
                      </div>
                    );
                  })}
                </div>
              )}

              <div className="pt-3 border-t border-border">
                <Button type="submit" disabled={isLoading} className="w-full">
                  {isLoading ? "Calling Tool..." : result || error ? "Call Again" : "Call Tool"}
                </Button>
              </div>
            </form>
          </div>
        </div>

        {/* Right Column - Tool Result */}
        <div className="bg-background border border-border rounded-lg">
          <div className="border-b border-border px-4 py-2">
            <h3 className="text-sm font-semibold text-foreground">Tool Result</h3>
          </div>

          <div className="p-4">
            {!result && !error && !isLoading ? (
              <div className="flex flex-col justify-center items-center h-48 text-muted-foreground">
                <div className="text-center max-w-sm">
                  <Zap className="mx-auto h-12 w-12 text-muted-foreground/50 mb-3" />
                  <h4 className="text-sm font-medium text-foreground mb-1">Ready to Call Tool</h4>
                  <p className="text-xs text-muted-foreground leading-relaxed">
                    Configure the input parameters and click &quot;Call Tool&quot; to see the results here.
                  </p>
                </div>
              </div>
            ) : (
              <div className="space-y-3">
                {result && !isLoading && !error && (
                  <div className="p-2 bg-green-50 dark:bg-green-950/30 border border-green-200 dark:border-green-900 rounded-lg">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-2">
                        <Check className="h-4 w-4 text-green-600 dark:text-green-400" />
                        <h4 className="text-xs font-medium text-green-900 dark:text-green-100">
                          Tool executed successfully
                        </h4>
                        {duration !== null && (
                          <span className="text-xs text-green-600 dark:text-green-400 ml-1">
                            • {(duration / 1000).toFixed(2)}s
                          </span>
                        )}
                      </div>

                      <div className="flex items-center space-x-1">
                        <div className="flex bg-background rounded border border-green-300 dark:border-green-800 p-0.5">
                          <button
                            type="button"
                            onClick={() => setViewMode("formatted")}
                            className={`px-2 py-1 text-xs font-medium rounded transition-colors ${
                              viewMode === "formatted"
                                ? "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-100"
                                : "text-green-600 dark:text-green-400 hover:text-green-800"
                            }`}
                          >
                            Formatted
                          </button>
                          <button
                            type="button"
                            onClick={() => setViewMode("json")}
                            className={`px-2 py-1 text-xs font-medium rounded transition-colors ${
                              viewMode === "json"
                                ? "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-100"
                                : "text-green-600 dark:text-green-400 hover:text-green-800"
                            }`}
                          >
                            JSON
                          </button>
                        </div>

                        <button
                          type="button"
                          onClick={handleCopyResult}
                          className="p-1 hover:bg-green-100 dark:hover:bg-green-900 rounded text-green-700 dark:text-green-300"
                          title="Copy response"
                        >
                          <Copy className="h-3.5 w-3.5" />
                        </button>
                      </div>
                    </div>
                  </div>
                )}

                <div className="max-h-96 overflow-y-auto">
                  {isLoading && (
                    <div className="flex flex-col justify-center items-center h-48 text-muted-foreground">
                      <div className="relative">
                        <div className="animate-spin rounded-full h-8 w-8 border-2 border-muted"></div>
                        <div className="animate-spin rounded-full h-8 w-8 border-2 border-primary border-t-transparent absolute top-0"></div>
                      </div>
                      <p className="text-sm font-medium mt-3 text-foreground">Calling tool...</p>
                      <p className="text-xs text-muted-foreground mt-1">
                        Please wait while we process your request
                      </p>
                    </div>
                  )}

                  {error && (
                    <div className="bg-destructive/5 border border-destructive/30 rounded-lg p-3">
                      <div className="flex items-start space-x-2">
                        <AlertCircle className="h-4 w-4 text-destructive flex-shrink-0 mt-0.5" />
                        <div className="flex-1">
                          <div className="flex items-center space-x-2 mb-1">
                            <h4 className="text-xs font-medium text-destructive">Tool Call Failed</h4>
                            {duration !== null && (
                              <span className="text-xs text-destructive">
                                • {(duration / 1000).toFixed(2)}s
                              </span>
                            )}
                          </div>
                          <div className="bg-background border border-destructive/30 rounded p-2 max-h-48 overflow-y-auto">
                            <pre className="text-xs whitespace-pre-wrap text-destructive font-mono">
                              {error.message}
                            </pre>
                          </div>
                        </div>
                      </div>
                    </div>
                  )}

                  {result && !isLoading && !error && (
                    <div className="space-y-3">
                      {viewMode === "formatted" ? (
                        result.map((content: any, idx: number) => (
                          <div key={idx} className="border border-border rounded-lg overflow-hidden">
                            {content.type === "text" && (
                              <div>
                                <div className="bg-muted px-3 py-1 border-b border-border">
                                  <span className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
                                    Text Response
                                  </span>
                                </div>
                                <div className="p-3">
                                  <div className="bg-background rounded border border-border max-h-64 overflow-y-auto">
                                    <div className="p-3 space-y-2">
                                      {content.text
                                        .split("\n\n")
                                        .map((section: string, sectionIndex: number) => {
                                          if (section.trim() === "") return null;

                                          if (section.startsWith("##")) {
                                            const headerText = section.replace(/^#+\s/, "");
                                            return (
                                              <div
                                                key={sectionIndex}
                                                className="border-b border-border pb-1 mb-2"
                                              >
                                                <h3 className="text-sm font-semibold text-foreground">
                                                  {headerText}
                                                </h3>
                                              </div>
                                            );
                                          }

                                          const urlRegex = /(https?:\/\/[^\s\)]+)/g;
                                          if (urlRegex.test(section)) {
                                            const parts = section.split(urlRegex);
                                            return (
                                              <div
                                                key={sectionIndex}
                                                className="bg-primary/5 border border-primary/20 rounded p-2"
                                              >
                                                <div className="text-xs text-foreground leading-relaxed whitespace-pre-wrap">
                                                  {parts.map((part, partIndex) => {
                                                    if (urlRegex.test(part)) {
                                                      return (
                                                        <a
                                                          key={partIndex}
                                                          href={part}
                                                          target="_blank"
                                                          rel="noopener noreferrer"
                                                          className="text-primary hover:text-primary/80 underline break-all"
                                                        >
                                                          {part}
                                                        </a>
                                                      );
                                                    }
                                                    return part;
                                                  })}
                                                </div>
                                              </div>
                                            );
                                          }

                                          if (section.includes("Score:")) {
                                            return (
                                              <div
                                                key={sectionIndex}
                                                className="bg-green-50 dark:bg-green-950/30 border-l-4 border-green-400 p-2 rounded-r"
                                              >
                                                <p className="text-xs text-green-800 dark:text-green-200 font-medium whitespace-pre-wrap">
                                                  {section}
                                                </p>
                                              </div>
                                            );
                                          }

                                          return (
                                            <div
                                              key={sectionIndex}
                                              className="bg-muted rounded p-2 border border-border"
                                            >
                                              <div className="text-xs text-foreground leading-relaxed whitespace-pre-wrap font-mono">
                                                {section}
                                              </div>
                                            </div>
                                          );
                                        })
                                        .filter(Boolean)}
                                    </div>
                                  </div>
                                </div>
                              </div>
                            )}

                            {content.type === "image" && content.url && (
                              <div>
                                <div className="bg-muted px-3 py-1 border-b border-border">
                                  <span className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
                                    Image Response
                                  </span>
                                </div>
                                <div className="p-3">
                                  <div className="bg-muted rounded p-3 border border-border">
                                    {/* eslint-disable-next-line @next/next/no-img-element */}
                                    <img
                                      src={content.url}
                                      alt="Tool result"
                                      className="max-w-full h-auto rounded shadow-sm"
                                    />
                                  </div>
                                </div>
                              </div>
                            )}

                            {content.type === "embedded_resource" && (
                              <div>
                                <div className="bg-muted px-3 py-1 border-b border-border">
                                  <span className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
                                    Embedded Resource
                                  </span>
                                </div>
                                <div className="p-3">
                                  <div className="flex items-center space-x-2 p-3 bg-primary/5 border border-primary/20 rounded">
                                    <InfoCircleOutlined className="h-5 w-5 text-primary flex-shrink-0" />
                                    <div className="flex-1">
                                      <p className="text-xs font-medium text-foreground">
                                        Resource Type: {content.resource_type}
                                      </p>
                                      {content.url && (
                                        <a
                                          href={content.url}
                                          target="_blank"
                                          rel="noopener noreferrer"
                                          className="inline-flex items-center text-xs text-primary hover:text-primary/80 hover:underline mt-1 transition-colors"
                                        >
                                          View Resource
                                        </a>
                                      )}
                                    </div>
                                  </div>
                                </div>
                              </div>
                            )}
                          </div>
                        ))
                      ) : (
                        <div className="bg-background rounded border border-border">
                          <div className="p-3 overflow-auto max-h-80 bg-muted">
                            <pre className="text-xs font-mono whitespace-pre-wrap break-all text-foreground">
                              {JSON.stringify(result, null, 2)}
                            </pre>
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
