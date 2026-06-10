import { useModelCostMap } from "@/app/(dashboard)/hooks/models/useModelCostMap";
import { ArrowRightIcon, PlayIcon, TrashIcon } from "@heroicons/react/outline";
import { Icon, Table, TableBody, TableCell, TableHead, TableHeaderCell, TableRow } from "@tremor/react";
import { Tooltip, Typography } from "antd";
import openai from "openai";
import React, { useEffect, useState } from "react";
import { Trans, useTranslation } from "react-i18next";
import { TFunction } from "i18next";
import DeleteResourceModal from "../../../common_components/DeleteResourceModal";
import { ProviderLogo } from "../../../molecules/models/ProviderLogo";
import NotificationsManager from "../../../molecules/notifications_manager";
import { getCallbacksCall, setCallbacksCall } from "../../../networking";
import { isProxyAdminRole } from "@/utils/roles";
import AddFallbacks from "./AddFallbacks";

type FallbackEntry = { [modelName: string]: string[] };
type Fallbacks = FallbackEntry[];

const modelCardClass =
  "inline-flex items-center gap-2 px-2.5 py-1 rounded-md border border-gray-200 bg-gray-50 text-sm font-medium text-gray-800 shrink-0";

function renderModelNameCell(modelName: string, getProviderFromModel?: (modelName: string) => string): React.ReactNode {
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
      <span className="inline-flex items-center justify-center w-8 h-8 shrink-0 self-start text-blue-600" aria-hidden>
        <ArrowRightIcon className="w-5 h-5 stroke-[2.5]" />
      </span>
      <span className="flex flex-wrap items-start gap-1 min-w-0">
        {list.map((model, i) => (
          <React.Fragment key={model}>
            {i > 0 && <Icon icon={ArrowRightIcon} size="xs" className="shrink-0 text-gray-400" />}
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

async function testFallbackModelResponse(selectedModel: string, accessToken: string, t: TFunction) {
  const isLocal = process.env.NODE_ENV === "development";
  if (isLocal != true) {
    console.log = function () {};
  }
  const proxyBaseUrl = isLocal ? "http://localhost:4000" : window.location.origin;
  const client = new openai.OpenAI({
    apiKey: accessToken,
    baseURL: proxyBaseUrl,
    dangerouslyAllowBrowser: true,
  });

  try {
    NotificationsManager.info(t("settingsPages.fallbacks.testingFallback"));

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
      <Trans
        i18nKey="settingsPages.fallbacks.testSuccess"
        values={{ selectedModel, responseModel: response.model }}
        components={{
          testModel: <strong />,
          receivedModel: <strong />,
          curlLink: (
            <a
              href="#"
              onClick={() => window.open("https://docs.litellm.ai/docs/proxy/reliability", "_blank")}
              style={{ textDecoration: "underline", color: "blue" }}
            />
          ),
        }}
      />,
    );
  } catch (error) {
    NotificationsManager.fromBackend(t("settingsPages.fallbacks.testError", { error }));
  }
}

const Fallbacks: React.FC<FallbacksProps> = ({ accessToken, userRole, userID, modelData }) => {
  const { t } = useTranslation();
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
      NotificationsManager.success(t("settingsPages.fallbacks.updateSuccess"));
    } catch (error) {
      NotificationsManager.fromBackend(t("settingsPages.fallbacks.updateFailed", { error }));
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
      NotificationsManager.fromBackend(t("settingsPages.fallbacks.updateFailed", { error }));
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
  // Admin Viewer follows the read-parity rule: see fallbacks, no writes.
  const canModify = isProxyAdminRole(userRole ?? "");

  return (
    <>
      {canModify && (
        <AddFallbacks
          models={modelData?.data ? modelData.data.map((data: any) => data.model_name) : []}
          accessToken={accessToken || ""}
          value={routerSettings.fallbacks || []}
          onChange={handleFallbacksChange}
        />
      )}
      {!hasFallbacks ? (
        <div className="rounded-lg border border-gray-200 bg-gray-50 px-4 py-6 text-center">
          <Typography.Text type="secondary">{t("settingsPages.fallbacks.noFallbacksConfigured")}</Typography.Text>
        </div>
      ) : (
        <Table>
          <TableHead>
            <TableRow>
              <TableHeaderCell>{t("settingsPages.fallbacks.modelNameColumn")}</TableHeaderCell>
              <TableHeaderCell>{t("settingsPages.fallbacks.fallbacksColumn")}</TableHeaderCell>
              <TableHeaderCell>{t("common.actions")}</TableHeaderCell>
            </TableRow>
          </TableHead>

          <TableBody>
            {routerSettings["fallbacks"].map((item: FallbackEntry, index: number) =>
              Object.entries(item).map(([key, value]) => (
                <TableRow key={index.toString() + key}>
                  <TableCell className="align-top">{renderModelNameCell(key, getProviderFromModel)}</TableCell>
                  <TableCell className="align-top">
                    {renderFallbacksChain(key, Array.isArray(value) ? value : [], getProviderFromModel)}
                  </TableCell>
                  <TableCell className="align-top">
                    {canModify && (
                      <>
                        <Tooltip title={t("settingsPages.fallbacks.testFallbackTooltip")}>
                          <Icon
                            icon={PlayIcon}
                            size="sm"
                            onClick={() => testFallbackModelResponse(Object.keys(item)[0], accessToken || "", t)}
                            className="cursor-pointer hover:text-blue-600"
                          />
                        </Tooltip>
                        <Tooltip title={t("settingsPages.fallbacks.deleteFallbackTooltip")}>
                          <span
                            data-testid="delete-fallback-button"
                            role="button"
                            tabIndex={0}
                            onClick={() => handleDeleteClick(item)}
                            onKeyDown={(e) => e.key === "Enter" && handleDeleteClick(item)}
                            className="cursor-pointer inline-flex"
                          >
                            <Icon icon={TrashIcon} size="sm" className="hover:text-red-600" />
                          </span>
                        </Tooltip>
                      </>
                    )}
                  </TableCell>
                </TableRow>
              )),
            )}
          </TableBody>
        </Table>
      )}
      <DeleteResourceModal
        isOpen={isDeleteModalOpen}
        title={t("settingsPages.fallbacks.deleteModalTitle")}
        message={t("settingsPages.fallbacks.deleteModalMessage")}
        resourceInformationTitle={t("settingsPages.fallbacks.deleteModalResourceTitle")}
        resourceInformation={[
          {
            label: t("settingsPages.fallbacks.modelNameColumn"),
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
