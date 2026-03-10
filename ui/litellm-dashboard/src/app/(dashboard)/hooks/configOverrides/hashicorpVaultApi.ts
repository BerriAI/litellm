import { getProxyBaseUrl, getGlobalLitellmHeaderName, deriveErrorMessage } from "@/components/networking";

export const getHashicorpVaultConfig = async (accessToken: string) => {
  const proxyBaseUrl = getProxyBaseUrl();
  const url = proxyBaseUrl
    ? `${proxyBaseUrl}/config_overrides/hashicorp_vault`
    : `/config_overrides/hashicorp_vault`;
  const response = await fetch(url, {
    method: "GET",
    headers: {
      [getGlobalLitellmHeaderName()]: `Bearer ${accessToken}`,
    },
  });
  if (!response.ok) {
    const errorData = await response.json();
    const errorMessage = deriveErrorMessage(errorData);
    throw new Error(errorMessage);
  }
  const data = await response.json();
  return data;
};

export const updateHashicorpVaultConfig = async (
  accessToken: string,
  config: Record<string, any>,
) => {
  const proxyBaseUrl = getProxyBaseUrl();
  const url = proxyBaseUrl
    ? `${proxyBaseUrl}/config_overrides/hashicorp_vault`
    : `/config_overrides/hashicorp_vault`;
  const response = await fetch(url, {
    method: "POST",
    headers: {
      [getGlobalLitellmHeaderName()]: `Bearer ${accessToken}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(config),
  });
  if (!response.ok) {
    const errorData = await response.json();
    const errorMessage = deriveErrorMessage(errorData);
    throw new Error(errorMessage);
  }
  const data = await response.json();
  return data;
};

export const deleteHashicorpVaultConfig = async (accessToken: string) => {
  const proxyBaseUrl = getProxyBaseUrl();
  const url = proxyBaseUrl
    ? `${proxyBaseUrl}/config_overrides/hashicorp_vault`
    : `/config_overrides/hashicorp_vault`;
  const response = await fetch(url, {
    method: "DELETE",
    headers: {
      [getGlobalLitellmHeaderName()]: `Bearer ${accessToken}`,
    },
  });
  if (!response.ok) {
    const errorData = await response.json();
    const errorMessage = deriveErrorMessage(errorData);
    throw new Error(errorMessage);
  }
  const data = await response.json();
  return data;
};

export const testHashicorpVaultConnection = async (accessToken: string) => {
  const proxyBaseUrl = getProxyBaseUrl();
  const url = proxyBaseUrl
    ? `${proxyBaseUrl}/config_overrides/hashicorp_vault/test_connection`
    : `/config_overrides/hashicorp_vault/test_connection`;
  const response = await fetch(url, {
    method: "POST",
    headers: {
      [getGlobalLitellmHeaderName()]: `Bearer ${accessToken}`,
    },
  });
  if (!response.ok) {
    const errorData = await response.json();
    const errorMessage = deriveErrorMessage(errorData);
    throw new Error(errorMessage);
  }
  const data = await response.json();
  return data;
};
