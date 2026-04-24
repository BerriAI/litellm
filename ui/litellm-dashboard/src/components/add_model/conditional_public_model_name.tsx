import React, { useEffect } from "react";
import { Controller, useFormContext, useWatch } from "react-hook-form";
import { Input } from "@/components/ui/input";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Label } from "@/components/ui/label";
import { Tooltip } from "../atoms/index";
import { Providers } from "../provider_info_helpers";

interface ModelMapping {
  public_name: string;
  litellm_model: string;
}

const ConditionalPublicModelName: React.FC = () => {
  const { control, setValue, getValues, formState } = useFormContext();

  const modelValue = useWatch({ control, name: "model" });
  const customModelName = useWatch({ control, name: "custom_model_name" });
  const selectedProvider = useWatch({ control, name: "custom_llm_provider" });
  const modelMappings = (useWatch({ control, name: "model_mappings" }) ??
    []) as ModelMapping[];

  const selectedModels = Array.isArray(modelValue)
    ? (modelValue as string[])
    : modelValue
      ? [modelValue as string]
      : [];
  const showPublicModelName = !selectedModels.includes("all-wildcard");

  useEffect(() => {
    if (customModelName && selectedModels.includes("custom")) {
      const currentMappings = (getValues("model_mappings") ||
        []) as ModelMapping[];
      const updatedMappings = currentMappings.map((mapping) => {
        if (
          mapping.public_name === "custom" ||
          mapping.litellm_model === "custom"
        ) {
          if (selectedProvider === Providers.Azure) {
            return {
              public_name: customModelName as string,
              litellm_model: `azure/${customModelName as string}`,
            };
          }
          return {
            public_name: customModelName as string,
            litellm_model: customModelName as string,
          };
        }
        return mapping;
      });
      setValue("model_mappings", updatedMappings);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [customModelName, JSON.stringify(selectedModels), selectedProvider]);

  useEffect(() => {
    if (
      selectedModels.length > 0 &&
      !selectedModels.includes("all-wildcard")
    ) {
      const currentMappings = (getValues("model_mappings") ||
        []) as ModelMapping[];

      const shouldUpdateMappings =
        currentMappings.length !== selectedModels.length ||
        !selectedModels.every((model) =>
          currentMappings.some((mapping) => {
            if (model === "custom") {
              return (
                mapping.litellm_model === "custom" ||
                mapping.litellm_model === customModelName
              );
            }
            if (selectedProvider === Providers.Azure) {
              return mapping.litellm_model === `azure/${model}`;
            }
            return mapping.litellm_model === model;
          }),
        );

      if (shouldUpdateMappings) {
        const mappings = selectedModels.map((model: string) => {
          if (model === "custom" && customModelName) {
            if (selectedProvider === Providers.Azure) {
              return {
                public_name: customModelName as string,
                litellm_model: `azure/${customModelName as string}`,
              };
            }
            return {
              public_name: customModelName as string,
              litellm_model: customModelName as string,
            };
          }
          if (selectedProvider === Providers.Azure) {
            return {
              public_name: model,
              litellm_model: `azure/${model}`,
            };
          }
          return {
            public_name: model,
            litellm_model: model,
          };
        });

        setValue("model_mappings", mappings);
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [JSON.stringify(selectedModels), customModelName, selectedProvider]);

  if (!showPublicModelName) return null;

  const publicNameTooltipContent = (
    <>
      <div className="mb-2 font-normal">
        The name you specify in your API calls to LiteLLM Proxy
      </div>
      <div className="mb-2 font-normal">
        <strong>Example:</strong> If you name your public model{" "}
        <code className="bg-foreground/70 text-background px-1 py-0.5 rounded text-xs">
          example-name
        </code>
        , and choose{" "}
        <code className="bg-foreground/70 text-background px-1 py-0.5 rounded text-xs">
          openai/qwen-plus-latest
        </code>{" "}
        as the LiteLLM model
      </div>
      <div className="mb-2 font-normal">
        <strong>Usage:</strong> You make an API call to the LiteLLM proxy with{" "}
        <code className="bg-foreground/70 text-background px-1 py-0.5 rounded text-xs">
          model = &quot;example-name&quot;
        </code>
      </div>
      <div className="font-normal">
        <strong>Result:</strong> LiteLLM sends{" "}
        <code className="bg-foreground/70 text-background px-1 py-0.5 rounded text-xs">
          qwen-plus-latest
        </code>{" "}
        to the provider
      </div>
    </>
  );

  const liteLLMModelTooltipContent = (
    <div>The model name LiteLLM will send to the LLM API</div>
  );

  const handlePublicNameChange = (index: number, newValue: string) => {
    const newMappings = [
      ...((getValues("model_mappings") || []) as ModelMapping[]),
    ];

    const isAnthropic = selectedProvider === Providers.Anthropic;
    const endsWith1m = newValue.endsWith("-1m");
    const litellmParams = getValues("litellm_extra_params") as
      | string
      | undefined;
    const isLitellmParamsEmpty = !litellmParams || litellmParams.trim() === "";

    let finalPublicName = newValue;

    if (isAnthropic && endsWith1m && isLitellmParamsEmpty) {
      const litellmParamsValue = JSON.stringify(
        { extra_headers: { "anthropic-beta": "context-1m-2025-08-07" } },
        null,
        2,
      );
      setValue("litellm_extra_params", litellmParamsValue);
      finalPublicName = newValue.slice(0, -3);
    }

    newMappings[index] = {
      ...newMappings[index],
      public_name: finalPublicName,
    };
    setValue("model_mappings", newMappings);
  };

  const mappingsError = (formState.errors as Record<string, { message?: string }>)
    .model_mappings;

  return (
    <div className="grid grid-cols-24 gap-2 mb-4 items-start">
      <Label
        className="col-span-10 pt-2"
        title="Map public model names to LiteLLM model names for load balancing"
      >
        Model Mappings
      </Label>
      <div className="col-span-14 space-y-2">
        <Controller
          control={control}
          name="model_mappings"
          rules={{
            validate: (value) => {
              if (!value || value.length === 0) {
                return "At least one model mapping is required";
              }
              const invalidMappings = (value as ModelMapping[]).filter(
                (mapping) =>
                  !mapping.public_name || mapping.public_name.trim() === "",
              );
              if (invalidMappings.length > 0) {
                return "All model mappings must have valid public names";
              }
              return true;
            },
          }}
          render={() => (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>
                    <span className="flex items-center">
                      Public Model Name
                      <Tooltip content={publicNameTooltipContent} width="500px" />
                    </span>
                  </TableHead>
                  <TableHead>
                    <span className="flex items-center">
                      LiteLLM Model Name
                      <Tooltip
                        content={liteLLMModelTooltipContent}
                        width="360px"
                      />
                    </span>
                  </TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {modelMappings.length === 0 ? (
                  <TableRow>
                    <TableCell
                      colSpan={2}
                      className="text-center text-muted-foreground py-4"
                    >
                      Select at least one model
                    </TableCell>
                  </TableRow>
                ) : (
                  modelMappings.map((record, index) => (
                    <TableRow key={`${record.litellm_model}-${index}`}>
                      <TableCell>
                        <Input
                          value={record.public_name}
                          onChange={(e) =>
                            handlePublicNameChange(index, e.target.value)
                          }
                        />
                      </TableCell>
                      <TableCell>{record.litellm_model}</TableCell>
                    </TableRow>
                  ))
                )}
              </TableBody>
            </Table>
          )}
        />
        {mappingsError?.message && (
          <p className="text-sm text-destructive">
            {String(mappingsError.message)}
          </p>
        )}
      </div>
    </div>
  );
};

export default ConditionalPublicModelName;
