import { getProxyUISettings } from "@/components/networking";

export const fetchProxySettings = async (accessToken: string | null) => {
  if (!accessToken) return null;

  try {
    const proxySettings = await getProxyUISettings(accessToken);
    return proxySettings;
  } catch (error) {
    console.error("Error fetching proxy settings:", error);
    return null;
  }
};
