import { useState, useCallback } from "react";
import { getProxyBaseUrl, getGlobalLitellmHeaderName } from "@/components/networking";
import NotificationsManager from "../molecules/notifications_manager";
import { MarginConfig } from "./types";
import { getProviderBackendValue } from "./provider_display_helpers";
import { Providers } from "../provider_info_helpers";

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
    value: number | { percentage?: number; fixed_amount?: number }
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
      const url = proxyBaseUrl 
        ? `${proxyBaseUrl}/config/cost_margin_config` 
        : "/config/cost_margin_config";
      
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
      NotificationsManager.fromBackend("Failed to fetch margin configuration");
    }
  }, [accessToken]);

  const saveMarginConfig = useCallback(async (config: MarginConfig) => {
    try {
      const proxyBaseUrl = getProxyBaseUrl();
      const url = proxyBaseUrl 
        ? `${proxyBaseUrl}/config/cost_margin_config` 
        : "/config/cost_margin_config";
      
      const response = await fetch(url, {
        method: "PATCH",
        headers: {
          [getGlobalLitellmHeaderName()]: `Bearer ${accessToken}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify(config),
      });

      if (response.ok) {
        NotificationsManager.success("Margin configuration updated successfully");
        await fetchMarginConfig();
      } else {
        const errorData = await response.json();
        const errorMessage = errorData.detail?.error || errorData.detail || "Failed to update settings";
        NotificationsManager.fromBackend(errorMessage);
      }
    } catch (error) {
      console.error("Error updating margin config:", error);
      NotificationsManager.fromBackend("Failed to update margin configuration");
    }
  }, [accessToken, fetchMarginConfig]);

  const handleAddMargin = useCallback(async (params: AddMarginParams): Promise<boolean> => {
    const { selectedProvider, marginType, percentageValue, fixedAmountValue } = params;

    if (!selectedProvider) {
      NotificationsManager.fromBackend("Please select a provider");
      return false;
    }

    let providerValue: string;
    if (selectedProvider === "global") {
      providerValue = "global";
    } else {
      const backendValue = getProviderBackendValue(selectedProvider);
      if (!backendValue) {
        NotificationsManager.fromBackend("Invalid provider selected");
        return false;
      }
      providerValue = backendValue;
    }

    if (marginConfig[providerValue]) {
      const displayName = providerValue === "global" ? "Global" : Providers[selectedProvider as keyof typeof Providers];
      NotificationsManager.fromBackend(
        `Margin for ${displayName} already exists. Edit it in the table above.`
      );
      return false;
    }

    let marginValue: number | { fixed_amount?: number };
    if (marginType === "percentage") {
      const percentValue = parseFloat(percentageValue);
      if (isNaN(percentValue) || percentValue < 0 || percentValue > 1000) {
        NotificationsManager.fromBackend("Percentage must be between 0% and 1000%");
        return false;
      }
      marginValue = percentValue / 100;
    } else {
      const fixedValue = parseFloat(fixedAmountValue);
      if (isNaN(fixedValue) || fixedValue < 0) {
        NotificationsManager.fromBackend("Fixed amount must be non-negative");
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
  }, [marginConfig, saveMarginConfig]);

  const handleRemoveMargin = useCallback(async (provider: string) => {
    const updatedConfig = { ...marginConfig };
    delete updatedConfig[provider];
    setMarginConfig(updatedConfig);
    await saveMarginConfig(updatedConfig);
  }, [marginConfig, saveMarginConfig]);

  const handleMarginChange = useCallback(async (
    provider: string,
    value: number | { percentage?: number; fixed_amount?: number }
  ) => {
    const updatedConfig = {
      ...marginConfig,
      [provider]: value,
    };
    setMarginConfig(updatedConfig);
    await saveMarginConfig(updatedConfig);
  }, [marginConfig, saveMarginConfig]);

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

