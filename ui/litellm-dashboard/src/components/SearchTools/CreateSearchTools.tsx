import { isAdminRole } from "@/utils/roles";
import { Info as InfoCircleOutlined } from "lucide-react";
import { useQuery } from "@tanstack/react-query";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import Image from "next/image";
import React, { useState } from "react";
import { Controller, FormProvider, useForm } from "react-hook-form";
import NotificationsManager from "../molecules/notifications_manager";
import { createSearchTool, fetchAvailableSearchProviders } from "../networking";
import SearchConnectionTest from "./SearchConnectionTest";
import { AvailableSearchProvider, SearchTool } from "./types";

// Search provider logos folder path (matches existing provider logo pattern)
const searchProviderLogosFolder = "../ui/assets/logos/";

// Helper function to get logo path for a search provider
const getSearchProviderLogo = (providerName: string): string => {
  return `${searchProviderLogosFolder}${providerName}.png`;
};

interface SearchProviderLabelProps {
  providerName: string;
  displayName: string;
}

const SearchProviderLabel: React.FC<SearchProviderLabelProps> = ({ providerName, displayName }) => (
  <div style={{ display: "flex", alignItems: "center" }}>
    <Image
      src={getSearchProviderLogo(providerName)}
      alt=""
      width={20}
      height={20}
      style={{
        marginRight: "8px",
        objectFit: "contain",
      }}
      onError={(e) => {
        e.currentTarget.style.display = "none";
      }}
    />
    <span>{displayName}</span>
  </div>
);

interface CreateSearchToolProps {
  userRole: string;
  accessToken: string | null;
  onCreateSuccess: (newSearchTool: SearchTool) => void;
  isModalVisible: boolean;
  setModalVisible: (visible: boolean) => void;
}

interface CreateFormValues {
  search_tool_name: string;
  search_provider: string;
  api_key?: string;
  api_base?: string;
  description?: string;
}

const SEARCH_TOOL_NAME_PATTERN = /^[a-zA-Z0-9_-]+$/;

const CreateSearchTool: React.FC<CreateSearchToolProps> = ({
  userRole,
  accessToken,
  onCreateSuccess,
  isModalVisible,
  setModalVisible,
}) => {
  const form = useForm<CreateFormValues>({
    defaultValues: {
      search_tool_name: "",
      search_provider: "",
      api_key: "",
      api_base: "",
      description: "",
    },
  });
  const [isLoading, setIsLoading] = useState(false);
  const [isTestModalVisible, setIsTestModalVisible] = useState(false);
  const [isTestingConnection, setIsTestingConnection] = useState(false);
  const [connectionTestId, setConnectionTestId] = useState<string>("");

  const {
    data: providersResponse,
    isLoading: isLoadingProviders,
  } = useQuery({
    queryKey: ["searchProviders"],
    queryFn: () => {
      if (!accessToken) throw new Error("Access Token required");
      return fetchAvailableSearchProviders(accessToken);
    },
    enabled: !!accessToken && isModalVisible,
  }) as { data: { providers: AvailableSearchProvider[] }; isLoading: boolean };

  const availableProviders = providersResponse?.providers || [];

  const onSubmit = form.handleSubmit(async (formValues) => {
    setIsLoading(true);
    try {
      const payload = {
        search_tool_name: formValues.search_tool_name,
        litellm_params: {
          search_provider: formValues.search_provider,
          api_key: formValues.api_key,
          api_base: formValues.api_base,
        },
        search_tool_info: formValues.description
          ? {
              description: formValues.description,
            }
          : undefined,
      };

      console.log(`Creating search tool with payload:`, payload);

      if (accessToken != null) {
        const response = await createSearchTool(accessToken, payload);
        NotificationsManager.success("Search tool created successfully");
        form.reset();
        setModalVisible(false);
        onCreateSuccess(response);
      }
    } catch (error) {
      NotificationsManager.error("Error creating search tool: " + error);
    } finally {
      setIsLoading(false);
    }
  });

  const handleCancel = () => {
    form.reset();
    setModalVisible(false);
  };

  const handleTestConnection = async () => {
    const valid = await form.trigger(["search_provider", "api_key"]);
    if (!valid) {
      NotificationsManager.error("Please fill in Search Provider and API Key before testing");
      return;
    }

    setIsTestingConnection(true);
    setConnectionTestId(`test-${Date.now()}`);
    setIsTestModalVisible(true);
  };

  // Clear form when modal closes to reset
  React.useEffect(() => {
    if (!isModalVisible) {
      form.reset();
    }
  }, [isModalVisible, form]);

  const formValues = form.watch();

  if (!isAdminRole(userRole)) {
    return null;
  }

  return (
    <Dialog open={isModalVisible} onOpenChange={(o) => (!o ? handleCancel() : undefined)}>
      <DialogContent className="max-w-[800px]">
        <DialogHeader>
          <DialogTitle asChild>
            <div className="flex items-center space-x-3 pb-4 border-b border-border">
              <span className="text-2xl">🔍</span>
              <h2 className="text-xl font-semibold text-foreground">Add New Search Tool</h2>
            </div>
          </DialogTitle>
        </DialogHeader>

        <FormProvider {...form}>
          <form onSubmit={onSubmit} className="space-y-6">
            <div className="grid grid-cols-1 gap-6">
              <div className="space-y-2">
                <Label htmlFor="create-search-tool-name" className="flex items-center">
                  Search Tool Name <span className="text-destructive ml-1">*</span>
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <InfoCircleOutlined className="ml-2 h-4 w-4 text-muted-foreground hover:text-foreground cursor-help" />
                      </TooltipTrigger>
                      <TooltipContent>
                        A unique name to identify this search tool configuration
                        (e.g., &quot;perplexity-search&quot;, &quot;tavily-news-search&quot;).
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                </Label>
                <Input
                  id="create-search-tool-name"
                  placeholder="e.g., perplexity-search, my-tavily-tool"
                  {...form.register("search_tool_name", {
                    required: "Please enter a search tool name",
                    pattern: {
                      value: SEARCH_TOOL_NAME_PATTERN,
                      message: "Name can only contain letters, numbers, hyphens, and underscores",
                    },
                  })}
                />
                {form.formState.errors.search_tool_name && (
                  <p className="text-sm text-destructive">
                    {form.formState.errors.search_tool_name.message as string}
                  </p>
                )}
              </div>

              <div className="space-y-2">
                <Label htmlFor="create-search-provider" className="flex items-center">
                  Search Provider <span className="text-destructive ml-1">*</span>
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <InfoCircleOutlined className="ml-2 h-4 w-4 text-muted-foreground hover:text-foreground cursor-help" />
                      </TooltipTrigger>
                      <TooltipContent>
                        Select the search provider you want to use. Each provider has different capabilities and pricing.
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                </Label>
                <Controller
                  control={form.control}
                  name="search_provider"
                  rules={{ required: "Please select a search provider" }}
                  render={({ field }) => (
                    <Select value={field.value} onValueChange={field.onChange} disabled={isLoadingProviders}>
                      <SelectTrigger id="create-search-provider">
                        <SelectValue placeholder="Select a search provider" />
                      </SelectTrigger>
                      <SelectContent>
                        {availableProviders.map((provider) => (
                          <SelectItem key={provider.provider_name} value={provider.provider_name}>
                            <SearchProviderLabel
                              providerName={provider.provider_name}
                              displayName={provider.ui_friendly_name}
                            />
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  )}
                />
                {form.formState.errors.search_provider && (
                  <p className="text-sm text-destructive">
                    {form.formState.errors.search_provider.message as string}
                  </p>
                )}
              </div>

              <div className="space-y-2">
                <Label htmlFor="create-api-key" className="flex items-center">
                  API Key
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <InfoCircleOutlined className="ml-2 h-4 w-4 text-muted-foreground hover:text-foreground cursor-help" />
                      </TooltipTrigger>
                      <TooltipContent>
                        The API key for authenticating with the search provider. This will be securely stored.
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                </Label>
                <Input
                  id="create-api-key"
                  type="password"
                  placeholder="Enter your API key"
                  {...form.register("api_key")}
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="create-description">Description (Optional)</Label>
                <Textarea
                  id="create-description"
                  rows={3}
                  placeholder="Brief description of this search tool's purpose"
                  {...form.register("description")}
                />
              </div>
            </div>

            <DialogFooter className="pt-6 border-t border-border flex !justify-between">
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <a
                      href="https://github.com/BerriAI/litellm/issues"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-primary hover:underline text-sm"
                    >
                      Need Help?
                    </a>
                  </TooltipTrigger>
                  <TooltipContent>Get help on our github</TooltipContent>
                </Tooltip>
              </TooltipProvider>
              <div className="space-x-2">
                <Button
                  type="button"
                  variant="outline"
                  onClick={handleTestConnection}
                  disabled={isTestingConnection}
                >
                  {isTestingConnection ? "Testing..." : "Test Connection"}
                </Button>
                <Button type="submit" disabled={isLoading}>
                  {isLoading ? "Adding..." : "Add Search Tool"}
                </Button>
              </div>
            </DialogFooter>
          </form>
        </FormProvider>

        {/* Test Connection Results Dialog */}
        <Dialog
          open={isTestModalVisible}
          onOpenChange={(o) => {
            if (!o) {
              setIsTestModalVisible(false);
              setIsTestingConnection(false);
            }
          }}
        >
          <DialogContent className="max-w-[700px]">
            <DialogHeader>
              <DialogTitle>Connection Test Results</DialogTitle>
            </DialogHeader>
            {isTestModalVisible && accessToken && (
              <SearchConnectionTest
                key={connectionTestId}
                litellmParams={{
                  search_provider: formValues.search_provider,
                  api_key: formValues.api_key,
                  api_base: formValues.api_base,
                }}
                accessToken={accessToken}
                onTestComplete={() => setIsTestingConnection(false)}
              />
            )}
            <DialogFooter>
              <Button
                variant="outline"
                onClick={() => {
                  setIsTestModalVisible(false);
                  setIsTestingConnection(false);
                }}
              >
                Close
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </DialogContent>
    </Dialog>
  );
};

export default CreateSearchTool;
