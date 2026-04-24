import React, { useEffect, useState } from "react";
import {
  Controller,
  FormProvider,
  UseFormReturn,
  useFormContext,
} from "react-hook-form";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { X, Loader2 } from "lucide-react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  RadioGroup,
  RadioGroupItem,
} from "@/components/ui/radio-group";
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { modelAvailableCall } from "../networking";
import ConnectionErrorDisplay from "./model_connection_test";
import { all_admin_roles } from "@/utils/roles";
import { handleAddAutoRouterSubmit } from "./handle_add_auto_router_submit";
import {
  fetchAvailableModels,
  ModelGroup,
} from "../playground/llm_calls/fetch_models";
import RouterConfigBuilder from "./RouterConfigBuilder";
import ComplexityRouterConfig from "./ComplexityRouterConfig";
import NotificationManager from "../molecules/notifications_manager";
import { Zap as ThunderboltOutlined, GitBranch as BranchesOutlined } from "lucide-react";

export interface AutoRouterFormValues {
  auto_router_name: string;
  auto_router_default_model?: string;
  auto_router_embedding_model?: string;
  model_access_group?: string[];
  // populated at submit time only
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  auto_router_config?: any;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  complexity_router_config?: any;
  model_type?: "semantic_router" | "complexity_router";
  custom_llm_provider?: string;
  model?: string;
  api_key?: string;
  team_id?: string;
}

interface AddAutoRouterTabProps {
  form: UseFormReturn<AutoRouterFormValues>;
  handleOk: (values: AutoRouterFormValues) => void | Promise<void>;
  accessToken: string;
  userRole: string;
}

type RouterType = "complexity" | "semantic";

interface ComplexityTiers {
  SIMPLE: string;
  MEDIUM: string;
  COMPLEX: string;
  REASONING: string;
}

const AddAutoRouterTab: React.FC<AddAutoRouterTabProps> = ({
  form,
  handleOk,
  accessToken,
  userRole,
}) => {
  const [isResultModalVisible, setIsResultModalVisible] =
    useState<boolean>(false);
  const [isTestingConnection, setIsTestingConnection] =
    useState<boolean>(false);
  const [connectionTestId, setConnectionTestId] = useState<string>("");

  const [modelAccessGroups, setModelAccessGroups] = useState<string[]>([]);
  const [modelInfo, setModelInfo] = useState<ModelGroup[]>([]);
  const [routerType, setRouterType] = useState<RouterType>("complexity");

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const [routerConfig, setRouterConfig] = useState<any>(null);
  const [complexityTiers, setComplexityTiers] = useState<ComplexityTiers>({
    SIMPLE: "",
    MEDIUM: "",
    COMPLEX: "",
    REASONING: "",
  });

  useEffect(() => {
    const fetchModelAccessGroups = async () => {
      const response = await modelAvailableCall(
        accessToken,
        "",
        "",
        false,
        null,
        true,
        true,
      );
      setModelAccessGroups(
        response["data"].map((model: { id: string }) => model.id),
      );
    };
    fetchModelAccessGroups();
  }, [accessToken]);

  useEffect(() => {
    const loadModels = async () => {
      try {
        const uniqueModels = await fetchAvailableModels(accessToken);
        setModelInfo(uniqueModels);
      } catch (error) {
        console.error("Error fetching model info for auto router:", error);
      }
    };
    loadModels();
  }, [accessToken]);

  const isAdmin = all_admin_roles.includes(userRole);

  const handleTestConnection = async () => {
    setIsTestingConnection(true);
    setConnectionTestId(`test-${Date.now()}`);
    setIsResultModalVisible(true);
  };

  const handleAutoRouterSubmit = form.handleSubmit(async (currentFormValues) => {
    if (!currentFormValues.auto_router_name) {
      NotificationManager.fromBackend("Please enter an Auto Router Name");
      return;
    }

    if (routerType === "complexity") {
      const filledTiers = Object.values(complexityTiers).filter(Boolean);
      if (filledTiers.length === 0) {
        NotificationManager.fromBackend(
          "Please select at least one model for a complexity tier",
        );
        return;
      }

      const defaultModel =
        complexityTiers.MEDIUM ||
        complexityTiers.SIMPLE ||
        complexityTiers.COMPLEX ||
        complexityTiers.REASONING;

      const submitValues: AutoRouterFormValues = {
        ...currentFormValues,
        auto_router_default_model: defaultModel,
        model_type: "complexity_router",
        complexity_router_config: { tiers: complexityTiers },
      };

      await handleAddAutoRouterSubmit(submitValues, accessToken, form, () =>
        handleOk(submitValues),
      );
    } else {
      if (!currentFormValues.auto_router_default_model) {
        NotificationManager.fromBackend("Please select a Default Model");
        return;
      }

      if (
        !routerConfig ||
        !routerConfig.routes ||
        routerConfig.routes.length === 0
      ) {
        NotificationManager.fromBackend(
          "Please configure at least one route for the auto router",
        );
        return;
      }

      const invalidRoutes = routerConfig.routes.filter(
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        (route: any) =>
          !route.name || !route.description || route.utterances.length === 0,
      );

      if (invalidRoutes.length > 0) {
        NotificationManager.fromBackend(
          "Please ensure all routes have a target model, description, and at least one utterance",
        );
        return;
      }

      const submitValues: AutoRouterFormValues = {
        ...currentFormValues,
        auto_router_config: routerConfig,
        model_type: "semantic_router",
      };

      await handleAddAutoRouterSubmit(submitValues, accessToken, form, () =>
        handleOk(submitValues),
      );
    }
  });

  return (
    <FormProvider {...form}>
      <h2 className="text-2xl font-semibold mb-2">Add Auto Router</h2>
      <p className="text-muted-foreground mb-6">
        Create an auto router that automatically selects the best model based on
        request complexity or semantic matching.
      </p>

      <Card className="p-6 mb-4">
        <div className="mb-4">
          <Label className="text-sm font-medium mb-2 block">Router Type</Label>
          <RadioGroup
            value={routerType}
            onValueChange={(v) => setRouterType(v as RouterType)}
            className="w-full flex flex-col gap-4"
          >
            <label className="flex items-start gap-3 cursor-pointer">
              <RadioGroupItem value="complexity" className="mt-1" />
              <div>
                <div className="flex items-center gap-2">
                  {/* eslint-disable-next-line litellm-ui/no-raw-tailwind-colors */}
                  <ThunderboltOutlined className="text-yellow-500" />
                  <span className="font-medium">Complexity Router</span>
                  <Badge variant="secondary">Recommended</Badge>
                </div>
                <div className="text-xs text-muted-foreground ml-0 mt-1">
                  Automatically routes based on request complexity. No
                  training data needed — just pick 4 models and go.
                </div>
              </div>
            </label>
            <label className="flex items-start gap-3 cursor-pointer">
              <RadioGroupItem value="semantic" className="mt-1" />
              <div>
                <div className="flex items-center gap-2">
                  {/* eslint-disable-next-line litellm-ui/no-raw-tailwind-colors */}
                  <BranchesOutlined className="text-blue-500" />
                  <span className="font-medium">Semantic Router</span>
                </div>
                <div className="text-xs text-muted-foreground ml-0 mt-1">
                  Routes based on semantic similarity to example utterances.
                  Requires embedding model and training examples.
                </div>
              </div>
            </label>
          </RadioGroup>
        </div>
      </Card>

      <Card className="p-6">
        <form onSubmit={handleAutoRouterSubmit}>
          <AutoRouterNameField />

          {routerType === "complexity" ? (
            <div className="w-full mb-4">
              <ComplexityRouterConfig
                modelInfo={modelInfo}
                value={complexityTiers}
                onChange={(tiers) => setComplexityTiers(tiers)}
              />
            </div>
          ) : (
            <>
              <div className="w-full mb-4">
                <RouterConfigBuilder
                  modelInfo={modelInfo}
                  value={routerConfig}
                  onChange={(config) => {
                    setRouterConfig(config);
                  }}
                />
              </div>

              <div className="grid grid-cols-24 gap-2 mb-4 items-start">
                <Label
                  className="col-span-10 pt-2"
                  title="Fallback model to use when auto routing logic cannot determine the best model"
                >
                  Default Model{" "}
                  <span className="text-destructive">*</span>
                </Label>
                <div className="col-span-14">
                  <Controller
                    control={form.control}
                    name="auto_router_default_model"
                    rules={{
                      required:
                        routerType === "semantic"
                          ? "Default model is required"
                          : false,
                    }}
                    render={({ field }) => (
                      <Select
                        value={(field.value as string) || ""}
                        onValueChange={field.onChange}
                      >
                        <SelectTrigger>
                          <SelectValue placeholder="Select a default model" />
                        </SelectTrigger>
                        <SelectContent>
                          {Array.from(
                            new Set(
                              modelInfo.map((option) => option.model_group),
                            ),
                          ).map((model_group) => (
                            <SelectItem key={model_group} value={model_group}>
                              {model_group}
                            </SelectItem>
                          ))}
                          <SelectItem value="custom">
                            Enter custom model name
                          </SelectItem>
                        </SelectContent>
                      </Select>
                    )}
                  />
                </div>
              </div>

              <div className="grid grid-cols-24 gap-2 mb-4 items-start">
                <Label
                  className="col-span-10 pt-2"
                  title="Optional: Embedding model to use for semantic routing decisions"
                >
                  Embedding Model
                </Label>
                <div className="col-span-14">
                  <Controller
                    control={form.control}
                    name="auto_router_embedding_model"
                    render={({ field }) => (
                      <Select
                        value={(field.value as string) || ""}
                        onValueChange={field.onChange}
                      >
                        <SelectTrigger>
                          <SelectValue placeholder="Select an embedding model (optional)" />
                        </SelectTrigger>
                        <SelectContent>
                          {Array.from(
                            new Set(
                              modelInfo.map((option) => option.model_group),
                            ),
                          ).map((model_group) => (
                            <SelectItem key={model_group} value={model_group}>
                              {model_group}
                            </SelectItem>
                          ))}
                          <SelectItem value="custom">
                            Enter custom model name
                          </SelectItem>
                        </SelectContent>
                      </Select>
                    )}
                  />
                </div>
              </div>
            </>
          )}

          <div className="flex items-center my-4">
            <div className="flex-grow border-t border-border"></div>
            <span className="px-4 text-muted-foreground text-sm">
              Additional Settings
            </span>
            <div className="flex-grow border-t border-border"></div>
          </div>

          {isAdmin && (
            <div className="grid grid-cols-24 gap-2 mb-4 items-start">
              <Label
                className="col-span-10 pt-2"
                title="Use model access groups to control who can access this auto router"
              >
                Model Access Group
              </Label>
              <div className="col-span-14">
                <Controller
                  control={form.control}
                  name="model_access_group"
                  defaultValue={[]}
                  render={({ field }) => (
                    <GroupTagInput
                      value={(field.value as string[]) ?? []}
                      onChange={field.onChange}
                      options={modelAccessGroups}
                    />
                  )}
                />
              </div>
            </div>
          )}

          <div className="flex justify-between items-center mb-4">
            <a
              href="https://github.com/BerriAI/litellm/issues"
              title="Get help on our github"
              target="_blank"
              rel="noopener noreferrer"
              className="text-primary hover:text-primary/80 underline"
            >
              Need Help?
            </a>
            <div className="space-x-2">
              <Button
                type="button"
                variant="outline"
                onClick={handleTestConnection}
                disabled={isTestingConnection}
              >
                {isTestingConnection && (
                  <Loader2 className="h-4 w-4 animate-spin mr-2" />
                )}
                Test Connection
              </Button>
              <Button type="submit">Add Auto Router</Button>
            </div>
          </div>
        </form>
      </Card>

      {/* Test Connection Results Modal */}
      <Dialog
        open={isResultModalVisible}
        onOpenChange={(open) => {
          if (!open) {
            setIsResultModalVisible(false);
            setIsTestingConnection(false);
          }
        }}
      >
        <DialogContent className="sm:max-w-[700px]">
          <DialogHeader>
            <DialogTitle>Connection Test Results</DialogTitle>
          </DialogHeader>
          {isResultModalVisible && (
            <ConnectionErrorDisplay
              key={connectionTestId}
              formValues={form.getValues()}
              accessToken={accessToken}
              testMode="chat"
              modelName={form.getValues("auto_router_name")}
              onClose={() => {
                setIsResultModalVisible(false);
                setIsTestingConnection(false);
              }}
              onTestComplete={() => setIsTestingConnection(false)}
            />
          )}
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => {
                setIsResultModalVisible(false);
                setIsTestingConnection(false);
              }}
            >
              <X className="h-4 w-4 mr-1" />
              Close
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </FormProvider>
  );
};

function AutoRouterNameField() {
  const { control, formState } = useFormContext<AutoRouterFormValues>();
  const error = (formState.errors as Record<string, { message?: string }>)
    .auto_router_name;
  return (
    <div className="grid grid-cols-24 gap-2 mb-4 items-start">
      <Label
        className="col-span-10 pt-2"
        title="Unique name for this auto router configuration"
      >
        Auto Router Name <span className="text-destructive">*</span>
      </Label>
      <div className="col-span-14 space-y-1">
        <Controller
          control={control}
          name="auto_router_name"
          rules={{ required: "Auto router name is required" }}
          render={({ field }) => (
            <Input
              placeholder="e.g., smart_router, auto_router_1"
              value={(field.value as string) ?? ""}
              onChange={(e) => field.onChange(e.target.value)}
            />
          )}
        />
        {error?.message && (
          <p className="text-sm text-destructive">{String(error.message)}</p>
        )}
      </div>
    </div>
  );
}

function GroupTagInput({
  value,
  onChange,
  options,
}: {
  value: string[];
  onChange: (next: string[]) => void;
  options: string[];
}) {
  const [input, setInput] = React.useState("");
  const remaining = options.filter((o) => !value.includes(o));

  const addValue = (next: string) => {
    const trimmed = next.trim();
    if (!trimmed || value.includes(trimmed)) return;
    onChange([...value, trimmed]);
  };

  return (
    <div className="space-y-2">
      <div className="flex gap-2">
        <Input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Type a group name and press Enter"
          onKeyDown={(e) => {
            if (e.key === "Enter" || e.key === ",") {
              e.preventDefault();
              addValue(input);
              setInput("");
            }
          }}
        />
        <Select
          value=""
          onValueChange={(v) => {
            if (v) addValue(v);
          }}
        >
          <SelectTrigger className="w-48">
            <SelectValue placeholder="Pick existing" />
          </SelectTrigger>
          <SelectContent>
            {remaining.length === 0 ? (
              <div className="py-2 px-3 text-sm text-muted-foreground">
                No options available
              </div>
            ) : (
              remaining.map((opt) => (
                <SelectItem key={opt} value={opt}>
                  {opt}
                </SelectItem>
              ))
            )}
          </SelectContent>
        </Select>
      </div>
      {value.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {value.map((v) => (
            <Badge
              key={v}
              variant="secondary"
              className="flex items-center gap-1"
            >
              {v}
              <button
                type="button"
                onClick={() => onChange(value.filter((s) => s !== v))}
                className="inline-flex items-center justify-center rounded-full hover:bg-muted-foreground/20"
                aria-label={`Remove ${v}`}
              >
                <X size={12} />
              </button>
            </Badge>
          ))}
        </div>
      )}
    </div>
  );
}

export default AddAutoRouterTab;
