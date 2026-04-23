import { useModelCostMap } from "@/app/(dashboard)/hooks/models/useModelCostMap";
import { ArrowRight, Play, Trash2 } from "lucide-react";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import openai from "openai";
import React, { useEffect, useState } from "react";
import DeleteResourceModal from "../../../common_components/DeleteResourceModal";
import { ProviderLogo } from "../../../molecules/models/ProviderLogo";
import NotificationsManager from "../../../molecules/notifications_manager";
import { getCallbacksCall, setCallbacksCall } from "../../../networking";
import AddFallbacks from "./AddFallbacks";

type FallbackEntry = { [modelName: string]: string[] };
type Fallbacks = FallbackEntry[];

const modelCardClass =
  "inline-flex items-center gap-2 px-2.5 py-1 rounded-md border border-border bg-muted text-sm font-medium text-foreground shrink-0";

function renderModelNameCell(
  modelName: string,
  getProviderFromModel?: (modelName: string) => string,
): React.ReactNode {
  const provider = getProviderFromModel?.(modelName) ?? modelName;
  return (
    <span className={modelCardClass}>
      <ProviderLogo provider={provider} className="w-4 h-4 shrink-0" />
      <span>{modelName}</span>
    </span>
  );
}

function renderFallbacksChain(
  _primaryModel: string,
  fallbackModels: string[],
  getProviderFromModel?: (modelName: string) => string,
): React.ReactNode {
  const list = Array.isArray(fallbackModels) ? fallbackModels : [];
  if (list.length === 0) return null;

  const ChainCard = ({ modelName }: { modelName: string }) => {
    const provider = getProviderFromModel?.(modelName) ?? modelName;
    return (
      <span className={modelCardClass}>
        <ProviderLogo provider={provider} className="w-4 h-4 shrink-0" />
        <span>{modelName}</span>
      </span>
    );
  };
  return (
    <span className="grid grid-cols-[auto_1fr] items-start gap-x-2 w-full min-w-0">
      <span
        className="inline-flex items-center justify-center w-8 h-8 shrink-0 self-start text-primary"
        aria-hidden
      >
        <ArrowRight className="w-5 h-5 stroke-[2.5]" />
      </span>
      <span className="flex flex-wrap items-start gap-1 min-w-0">
        {list.map((model, i) => (
          <React.Fragment key={model}>
            {i > 0 && (
              <ArrowRight className="w-3 h-3 shrink-0 text-muted-foreground" />
            )}
            <ChainCard modelName={model} />
          </React.Fragment>
        ))}
      </span>
    </span>
  );
}

interface FallbacksProps {
  accessToken: string | null;
  userRole: string | null;
  userID: string | null;
  modelData: any;
}

async function testFallbackModelResponse(selectedModel: string, accessToken: string) {
  const isLocal = process.env.NODE_ENV === "development";
  if (isLocal != true) {
    console.log = function () { };
  }
  const proxyBaseUrl = isLocal ? "http://localhost:4000" : window.location.origin;
  const client = new openai.OpenAI({
    apiKey: accessToken,
    baseURL: proxyBaseUrl,
    dangerouslyAllowBrowser: true,
  });

  try {
    NotificationsManager.info("Testing fallback model response...");

    const response = await client.chat.completions.create({
      model: selectedModel,
      messages: [
        {
          role: "user",
          content: "Hi, this is a test message",
        },
      ],
      // @ts-ignore
      mock_testing_fallbacks: true,
    });

    NotificationsManager.success(
      <span>
        Test model=<strong>{selectedModel}</strong>, received model=
        <strong>{response.model}</strong>. See{" "}
        <a
          href="#"
          onClick={() => window.open("https://docs.litellm.ai/docs/proxy/reliability", "_blank")}
          style={{ textDecoration: "underline", color: "blue" }}
        >
          curl
        </a>
      </span>,
    );
  } catch (error) {
    NotificationsManager.fromBackend(
      `Error occurred while generating model response. Please try again. Error: ${error}`,
    );
  }
}

const Fallbacks: React.FC<FallbacksProps> = ({ accessToken, userRole, userID, modelData }) => {
  const [routerSettings, setRouterSettings] = useState<{ [key: string]: any }>({});
  const [isDeleting, setIsDeleting] = useState(false);
  const [fallbackToDelete, setFallbackToDelete] = useState<FallbackEntry | null>(null);
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);

  const { data: modelCostMapData } = useModelCostMap();
  const getProviderFromModel = (model: string): string => {
    if (modelCostMapData != null && typeof modelCostMapData === "object" && model in modelCostMapData) {
      return modelCostMapData[model]["litellm_provider"] ?? "";
    }
    return "";
  };

  useEffect(() => {
    if (!accessToken || !userRole || !userID) {
      return;
    }
    getCallbacksCall(accessToken, userID, userRole).then((data) => {
      console.log("callbacks", data);
      let router_settings = data.router_settings;
      if ("model_group_retry_policy" in router_settings) {
        delete router_settings["model_group_retry_policy"];
      }
      setRouterSettings(router_settings);
    });
  }, [accessToken, userRole, userID]);

  const handleDeleteClick = (fallbackEntry: FallbackEntry) => {
    setFallbackToDelete(fallbackEntry);
    setIsDeleteModalOpen(true);
  };

  const handleDeleteConfirm = async () => {
    if (!fallbackToDelete || !accessToken) {
      return;
    }

    const key = Object.keys(fallbackToDelete)[0];
    if (!key) {
      return;
    }
    setIsDeleting(true);

    const updatedFallbacks = routerSettings["fallbacks"]
      .map((dict: FallbackEntry) => {
        const newDict = { ...dict };
        if (key in newDict && Array.isArray(newDict[key])) {
          delete newDict[key];
        }
        return newDict;
      })
      .filter((dict: FallbackEntry) => Object.keys(dict).length > 0);

    const updatedSettings = {
      ...routerSettings,
      fallbacks: updatedFallbacks,
    };

    const payload = {
      router_settings: updatedSettings,
    };

    try {
      await setCallbacksCall(accessToken, payload);
      setRouterSettings(updatedSettings);
      NotificationsManager.success("Router settings updated successfully");
    } catch (error) {
      NotificationsManager.fromBackend("Failed to update router settings: " + error);
    } finally {
      setIsDeleting(false);
      setIsDeleteModalOpen(false);
      setFallbackToDelete(null);
    }
  };

  const handleDeleteCancel = () => {
    setIsDeleteModalOpen(false);
    setFallbackToDelete(null);
  };

  if (!accessToken) {
    return null;
  }

  const handleFallbacksChange = async (fallbacks: Fallbacks): Promise<void> => {
    if (!accessToken) {
      return;
    }

    const updatedSettings = {
      ...routerSettings,
      fallbacks: fallbacks,
    };

    const payload = {
      router_settings: updatedSettings,
    };

    try {
      await setCallbacksCall(accessToken, payload);
      // Update UI only after successful API call
      setRouterSettings(updatedSettings);
    } catch (error) {
      // Revert on error by refetching from server
      NotificationsManager.fromBackend("Failed to update router settings: " + error);
      if (accessToken && userRole && userID) {
        getCallbacksCall(accessToken, userID, userRole).then((data) => {
          let router_settings = data.router_settings;
          if ("model_group_retry_policy" in router_settings) {
            delete router_settings["model_group_retry_policy"];
          }
          setRouterSettings(router_settings);
        });
      }
      // Re-throw error so caller can handle it
      throw error;
    }
  };

  const hasFallbacks = Array.isArray(routerSettings.fallbacks) && routerSettings.fallbacks.length > 0;

  return (
    <>
      <AddFallbacks
        models={modelData?.data ? modelData.data.map((data: any) => data.model_name) : []}
        accessToken={accessToken || ""}
        value={routerSettings.fallbacks || []}
        onChange={handleFallbacksChange}
      />
      {!hasFallbacks ? (
        <div className="rounded-lg border border-border bg-muted px-4 py-6 text-center">
          <span className="text-muted-foreground text-sm">
            No fallbacks configured. Add fallbacks to automatically try another
            model when the primary fails.
          </span>
        </div>
      ) : (
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Model Name</TableHead>
              <TableHead>Fallbacks</TableHead>
              <TableHead>Actions</TableHead>
            </TableRow>
          </TableHeader>

          <TableBody>
            {routerSettings["fallbacks"].map(
              (item: FallbackEntry, index: number) =>
                Object.entries(item).map(([key, value]) => (
                  <TableRow key={index.toString() + key}>
                    <TableCell className="align-top">
                      {renderModelNameCell(key, getProviderFromModel)}
                    </TableCell>
                    <TableCell className="align-top">
                      {renderFallbacksChain(
                        key,
                        Array.isArray(value) ? value : [],
                        getProviderFromModel,
                      )}
                    </TableCell>
                    <TableCell className="align-top">
                      <TooltipProvider>
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <button
                              type="button"
                              onClick={() =>
                                testFallbackModelResponse(
                                  Object.keys(item)[0],
                                  accessToken || "",
                                )
                              }
                              className="cursor-pointer hover:text-primary inline-flex p-1"
                              aria-label="Test fallback"
                            >
                              <Play className="h-4 w-4" />
                            </button>
                          </TooltipTrigger>
                          <TooltipContent>Test fallback</TooltipContent>
                        </Tooltip>
                      </TooltipProvider>
                      <TooltipProvider>
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <button
                              type="button"
                              data-testid="delete-fallback-button"
                              onClick={() => handleDeleteClick(item)}
                              onKeyDown={(e) =>
                                e.key === "Enter" && handleDeleteClick(item)
                              }
                              className="cursor-pointer inline-flex p-1 hover:text-destructive"
                              aria-label="Delete fallback"
                            >
                              <Trash2 className="h-4 w-4" />
                            </button>
                          </TooltipTrigger>
                          <TooltipContent>Delete fallback</TooltipContent>
                        </Tooltip>
                      </TooltipProvider>
                    </TableCell>
                  </TableRow>
                )),
            )}
          </TableBody>
        </Table>
      )}
      <DeleteResourceModal
        isOpen={isDeleteModalOpen}
        title="Delete Fallback?"
        message="Are you sure you want to delete this fallback? This action cannot be undone."
        resourceInformationTitle="Fallback Information"
        resourceInformation={[
          {
            label: "Model Name",
            value: fallbackToDelete ? Object.keys(fallbackToDelete)[0] : "",
            code: true,
          },
        ]}
        onCancel={handleDeleteCancel}
        onOk={handleDeleteConfirm}
        confirmLoading={isDeleting}
      />
    </>
  );
};

export default Fallbacks;
