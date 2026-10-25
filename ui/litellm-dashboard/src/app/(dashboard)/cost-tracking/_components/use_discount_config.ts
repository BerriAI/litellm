import { createElement, useState, useCallback } from "react";
import { getProxyBaseUrl, getGlobalLitellmHeaderName } from "@/components/networking";
import { toast } from "@/lib/toast";
import { DiscountConfig } from "./types";
import { getProviderBackendValue } from "./provider_display_helpers";
import { Providers } from "@/components/provider_info_helpers";
import TranslatedMessage from "@/components/shared/TranslatedMessage";

export interface UseDiscountConfigProps {
  accessToken: string | null;
}

export interface UseDiscountConfigReturn {
  discountConfig: DiscountConfig;
  setDiscountConfig: React.Dispatch<React.SetStateAction<DiscountConfig>>;
  fetchDiscountConfig: () => Promise<void>;
  saveDiscountConfig: (config: DiscountConfig) => Promise<void>;
  handleAddProvider: (
    selectedProvider: string | undefined,
    newDiscount: string,
    modelPattern: string,
  ) => Promise<boolean>;
  handleRemoveProvider: (provider: string) => Promise<void>;
  handleDiscountChange: (provider: string, value: string) => Promise<void>;
}

export function useDiscountConfig({ accessToken }: UseDiscountConfigProps): UseDiscountConfigReturn {
  const [discountConfig, setDiscountConfig] = useState<DiscountConfig>({});

  const fetchDiscountConfig = useCallback(async () => {
    try {
      const proxyBaseUrl = getProxyBaseUrl();
      const url = proxyBaseUrl ? `${proxyBaseUrl}/config/cost_discount_config` : "/config/cost_discount_config";

      const response = await fetch(url, {
        method: "GET",
        headers: {
          [getGlobalLitellmHeaderName()]: `Bearer ${accessToken}`,
          "Content-Type": "application/json",
        },
      });

      if (response.ok) {
        const data = await response.json();
        setDiscountConfig(data.values || {});
      } else {
        console.error("Failed to fetch discount config");
      }
    } catch (error) {
      console.error("Error fetching discount config:", error);
      // 固定提示在通知树中翻译，异步回调不捕获语言。
      toast.fromError(null, {
        description: createElement(TranslatedMessage, { message: "Failed to fetch discount configuration" }),
      });
    }
  }, [accessToken]);

  const saveDiscountConfig = useCallback(
    async (config: DiscountConfig) => {
      try {
        const proxyBaseUrl = getProxyBaseUrl();
        const url = proxyBaseUrl ? `${proxyBaseUrl}/config/cost_discount_config` : "/config/cost_discount_config";

        const response = await fetch(url, {
          method: "PATCH",
          headers: {
            [getGlobalLitellmHeaderName()]: `Bearer ${accessToken}`,
            "Content-Type": "application/json",
          },
          body: JSON.stringify(config),
        });

        if (response.ok) {
          toast.success(createElement(TranslatedMessage, { message: "Discount configuration updated successfully" }));
          await fetchDiscountConfig();
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
        console.error("Error updating discount config:", error);
        toast.fromError(null, {
          description: createElement(TranslatedMessage, { message: "Failed to update discount configuration" }),
        });
      }
    },
    [accessToken, fetchDiscountConfig],
  );

  const handleAddProvider = useCallback(
    async (selectedProvider: string | undefined, newDiscount: string, modelPattern: string): Promise<boolean> => {
      if (!selectedProvider || !newDiscount) {
        toast.fromError(null, {
          description: createElement(TranslatedMessage, {
            message: "Please select a provider and enter discount percentage",
          }),
        });
        return false;
      }

      const percentageValue = parseFloat(newDiscount);
      if (isNaN(percentageValue) || percentageValue < 0 || percentageValue > 100) {
        toast.fromError(null, {
          description: createElement(TranslatedMessage, { message: "Discount must be between 0% and 100%" }),
        });
        return false;
      }

      const trimmedPattern = modelPattern.trim();

      const providerValue = getProviderBackendValue(selectedProvider);

      if (!providerValue) {
        toast.fromError(null, {
          description: createElement(TranslatedMessage, { message: "Invalid provider selected" }),
        });
        return false;
      }

      const configKey = trimmedPattern ? `${providerValue}/${trimmedPattern}` : providerValue;

      if (configKey in discountConfig) {
        const displayName = Providers[selectedProvider as keyof typeof Providers];
        toast.fromError(null, {
          description: createElement(TranslatedMessage, {
            message: trimmedPattern
              ? "Discount for {provider} ({models}) already exists. Edit it in the table above."
              : "Discount for {provider} (all models) already exists. Edit it in the table above.",
            values: { provider: displayName, models: trimmedPattern },
          }),
        });
        return false;
      }

      const discountValue = percentageValue / 100;
      const updatedConfig = {
        ...discountConfig,
        [configKey]: discountValue,
      };

      setDiscountConfig(updatedConfig);
      await saveDiscountConfig(updatedConfig);
      return true;
    },
    [discountConfig, saveDiscountConfig],
  );

  const handleRemoveProvider = useCallback(
    async (provider: string) => {
      const updatedConfig = { ...discountConfig };
      delete updatedConfig[provider];
      setDiscountConfig(updatedConfig);
      await saveDiscountConfig(updatedConfig);
    },
    [discountConfig, saveDiscountConfig],
  );

  const handleDiscountChange = useCallback(
    async (provider: string, value: string) => {
      const discountValue = parseFloat(value);
      if (!isNaN(discountValue) && discountValue >= 0 && discountValue <= 1) {
        const updatedConfig = {
          ...discountConfig,
          [provider]: discountValue,
        };
        setDiscountConfig(updatedConfig);
        await saveDiscountConfig(updatedConfig);
      }
    },
    [discountConfig, saveDiscountConfig],
  );

  return {
    discountConfig,
    setDiscountConfig,
    fetchDiscountConfig,
    saveDiscountConfig,
    handleAddProvider,
    handleRemoveProvider,
    handleDiscountChange,
  };
}
