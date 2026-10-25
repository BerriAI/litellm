import { createElement, useState, useCallback } from "react";
import { getProxyBaseUrl, getGlobalLitellmHeaderName } from "@/components/networking";
import { toast } from "@/lib/toast";
import { MarginConfig } from "./types";
import { getProviderBackendValue } from "./provider_display_helpers";
import { Providers } from "@/components/provider_info_helpers";
import TranslatedMessage from "@/components/shared/TranslatedMessage";

export interface UseMarginConfigProps {
  accessToken: string | null;
}

export interface UseMarginConfigReturn {
  marginConfig: MarginConfig;
  setMarginConfig: React.Dispatch<React.SetStateAction<MarginConfig>>;
  fetchMarginConfig: () => Promise<void>;
  saveMarginConfig: (config: MarginConfig) => Promise<void>;
  handleAddMargin: (params: AddMarginParams) => Promise<boolean>;
  handleRemoveMargin: (provider: string) => Promise<void>;
  handleMarginChange: (
    provider: string,
    value: number | { percentage?: number; fixed_amount?: number },
  ) => Promise<void>;
}

export interface AddMarginParams {
  selectedProvider: string | undefined;
  marginType: "percentage" | "fixed";
  percentageValue: string;
  fixedAmountValue: string;
}

export function useMarginConfig({ accessToken }: UseMarginConfigProps): UseMarginConfigReturn {
  const [marginConfig, setMarginConfig] = useState<MarginConfig>({});

  const fetchMarginConfig = useCallback(async () => {
    try {
      const proxyBaseUrl = getProxyBaseUrl();
      const url = proxyBaseUrl ? `${proxyBaseUrl}/config/cost_margin_config` : "/config/cost_margin_config";

      const response = await fetch(url, {
        method: "GET",
        headers: {
          [getGlobalLitellmHeaderName()]: `Bearer ${accessToken}`,
          "Content-Type": "application/json",
        },
      });

      if (response.ok) {
        const data = await response.json();
        setMarginConfig(data.values || {});
      } else {
        console.error("Failed to fetch margin config");
      }
    } catch (error) {
      console.error("Error fetching margin config:", error);
      // 固定提示在通知树中翻译，异步回调不捕获语言。
      toast.fromError(null, {
        description: createElement(TranslatedMessage, { message: "Failed to fetch margin configuration" }),
      });
    }
  }, [accessToken]);

  const saveMarginConfig = useCallback(
    async (config: MarginConfig) => {
      try {
        const proxyBaseUrl = getProxyBaseUrl();
        const url = proxyBaseUrl ? `${proxyBaseUrl}/config/cost_margin_config` : "/config/cost_margin_config";

        const response = await fetch(url, {
          method: "PATCH",
          headers: {
            [getGlobalLitellmHeaderName()]: `Bearer ${accessToken}`,
            "Content-Type": "application/json",
          },
          body: JSON.stringify(config),
        });

        if (response.ok) {
          toast.success(createElement(TranslatedMessage, { message: "Margin configuration updated successfully" }));
          await fetchMarginConfig();
        } else {
          const errorData = await response.json();
          const errorMessage = errorData.detail?.error || errorData.detail;
          // 保留响应状态和原始业务消息，只有缺少消息时使用本地提示。
          toast.fromError(
            { response: { status: response.status, data: errorData } },
            errorMessage
              ? undefined
              : { description: createElement(TranslatedMessage, { message: "Failed to update settings" }) },
          );
        }
      } catch (error) {
        console.error("Error updating margin config:", error);
        toast.fromError(null, {
          description: createElement(TranslatedMessage, { message: "Failed to update margin configuration" }),
        });
      }
    },
    [accessToken, fetchMarginConfig],
  );

  const handleAddMargin = useCallback(
    async (params: AddMarginParams): Promise<boolean> => {
      const { selectedProvider, marginType, percentageValue, fixedAmountValue } = params;

      if (!selectedProvider) {
        toast.fromError(null, {
          description: createElement(TranslatedMessage, { message: "Please select a provider" }),
        });
        return false;
      }

      let providerValue: string;
      if (selectedProvider === "global") {
        providerValue = "global";
      } else {
        const backendValue = getProviderBackendValue(selectedProvider);
        if (!backendValue) {
          toast.fromError(null, {
            description: createElement(TranslatedMessage, { message: "Invalid provider selected" }),
          });
          return false;
        }
        providerValue = backendValue;
      }

      if (marginConfig[providerValue]) {
        toast.fromError(null, {
          description: createElement(TranslatedMessage, {
            message:
              providerValue === "global"
                ? "Margin for Global already exists. Edit it in the table above."
                : "Margin for {provider} already exists. Edit it in the table above.",
            values:
              providerValue === "global"
                ? undefined
                : { provider: Providers[selectedProvider as keyof typeof Providers] },
          }),
        });
        return false;
      }

      let marginValue: number | { fixed_amount?: number };
      if (marginType === "percentage") {
        const percentValue = parseFloat(percentageValue);
        if (isNaN(percentValue) || percentValue < 0 || percentValue > 1000) {
          toast.fromError(null, {
            description: createElement(TranslatedMessage, { message: "Percentage must be between 0% and 1000%" }),
          });
          return false;
        }
        marginValue = percentValue / 100;
      } else {
        const fixedValue = parseFloat(fixedAmountValue);
        if (isNaN(fixedValue) || fixedValue < 0) {
          toast.fromError(null, {
            description: createElement(TranslatedMessage, { message: "Fixed amount must be non-negative" }),
          });
          return false;
        }
        marginValue = { fixed_amount: fixedValue };
      }

      const updatedConfig = {
        ...marginConfig,
        [providerValue]: marginValue,
      };

      setMarginConfig(updatedConfig);
      await saveMarginConfig(updatedConfig);
      return true;
    },
    [marginConfig, saveMarginConfig],
  );

  const handleRemoveMargin = useCallback(
    async (provider: string) => {
      const updatedConfig = { ...marginConfig };
      delete updatedConfig[provider];
      setMarginConfig(updatedConfig);
      await saveMarginConfig(updatedConfig);
    },
    [marginConfig, saveMarginConfig],
  );

  const handleMarginChange = useCallback(
    async (provider: string, value: number | { percentage?: number; fixed_amount?: number }) => {
      const updatedConfig = {
        ...marginConfig,
        [provider]: value,
      };
      setMarginConfig(updatedConfig);
      await saveMarginConfig(updatedConfig);
    },
    [marginConfig, saveMarginConfig],
  );

  return {
    marginConfig,
    setMarginConfig,
    fetchMarginConfig,
    saveMarginConfig,
    handleAddMargin,
    handleRemoveMargin,
    handleMarginChange,
  };
}
