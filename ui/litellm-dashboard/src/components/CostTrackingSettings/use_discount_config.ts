import { useState, useCallback } from "react";
import { getProxyBaseUrl, getGlobalLitellmHeaderName } from "@/components/networking";
import NotificationsManager from "../molecules/notifications_manager";
import { DiscountConfig } from "./types";
import { getProviderBackendValue } from "./provider_display_helpers";
import { Providers } from "../provider_info_helpers";

export interface UseDiscountConfigProps {
  accessToken: string | null;
}

export interface UseDiscountConfigReturn {
  discountConfig: DiscountConfig;
  setDiscountConfig: React.Dispatch<React.SetStateAction<DiscountConfig>>;
  fetchDiscountConfig: () => Promise<void>;
  saveDiscountConfig: (config: DiscountConfig) => Promise<void>;
  handleAddProvider: (selectedProvider: string | undefined, newDiscount: string) => Promise<boolean>;
  handleRemoveProvider: (provider: string) => Promise<void>;
  handleDiscountChange: (provider: string, value: string) => Promise<void>;
}

export function useDiscountConfig({ accessToken }: UseDiscountConfigProps): UseDiscountConfigReturn {
  const [discountConfig, setDiscountConfig] = useState<DiscountConfig>({});

  const fetchDiscountConfig = useCallback(async () => {
    try {
      const proxyBaseUrl = getProxyBaseUrl();
      const url = proxyBaseUrl 
        ? `${proxyBaseUrl}/config/cost_discount_config` 
        : "/config/cost_discount_config";
      
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
      NotificationsManager.fromBackend("Failed to fetch discount configuration");
    }
  }, [accessToken]);

  const saveDiscountConfig = useCallback(async (config: DiscountConfig) => {
    try {
      const proxyBaseUrl = getProxyBaseUrl();
      const url = proxyBaseUrl 
        ? `${proxyBaseUrl}/config/cost_discount_config` 
        : "/config/cost_discount_config";
      
      const response = await fetch(url, {
        method: "PATCH",
        headers: {
          [getGlobalLitellmHeaderName()]: `Bearer ${accessToken}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify(config),
      });

      if (response.ok) {
        NotificationsManager.success("Discount configuration updated successfully");
        await fetchDiscountConfig();
      } else {
        const errorData = await response.json();
        const errorMessage = errorData.detail?.error || errorData.detail || "Failed to update settings";
        NotificationsManager.fromBackend(errorMessage);
      }
    } catch (error) {
      console.error("Error updating discount config:", error);
      NotificationsManager.fromBackend("Failed to update discount configuration");
    }
  }, [accessToken, fetchDiscountConfig]);

  const handleAddProvider = useCallback(async (
    selectedProvider: string | undefined,
    newDiscount: string
  ): Promise<boolean> => {
    if (!selectedProvider || !newDiscount) {
      NotificationsManager.fromBackend("Please select a provider and enter discount percentage");
      return false;
    }
    
    const percentageValue = parseFloat(newDiscount);
    if (isNaN(percentageValue) || percentageValue < 0 || percentageValue > 100) {
      NotificationsManager.fromBackend("Discount must be between 0% and 100%");
      return false;
    }

    const providerValue = getProviderBackendValue(selectedProvider);
    
    if (!providerValue) {
      NotificationsManager.fromBackend("Invalid provider selected");
      return false;
    }

    if (discountConfig[providerValue]) {
      NotificationsManager.fromBackend(
        `Discount for ${Providers[selectedProvider as keyof typeof Providers]} already exists. Edit it in the table above.`
      );
      return false;
    }

    const discountValue = percentageValue / 100;
    const updatedConfig = {
      ...discountConfig,
      [providerValue]: discountValue,
    };
    
    setDiscountConfig(updatedConfig);
    await saveDiscountConfig(updatedConfig);
    return true;
  }, [discountConfig, saveDiscountConfig]);

  const handleRemoveProvider = useCallback(async (provider: string) => {
    const updatedConfig = { ...discountConfig };
    delete updatedConfig[provider];
    setDiscountConfig(updatedConfig);
    await saveDiscountConfig(updatedConfig);
  }, [discountConfig, saveDiscountConfig]);

  const handleDiscountChange = useCallback(async (provider: string, value: string) => {
    const discountValue = parseFloat(value);
    if (!isNaN(discountValue) && discountValue >= 0 && discountValue <= 1) {
      const updatedConfig = {
        ...discountConfig,
        [provider]: discountValue,
      };
      setDiscountConfig(updatedConfig);
      await saveDiscountConfig(updatedConfig);
    }
  }, [discountConfig, saveDiscountConfig]);

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

