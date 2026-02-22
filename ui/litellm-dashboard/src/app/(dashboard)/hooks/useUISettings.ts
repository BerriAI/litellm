import { getUiSettings } from "@/components/networking";
import { useQuery } from "@tanstack/react-query";

export interface UISettingsData {
  disable_model_add_for_internal_users: boolean;
  disable_team_admin_delete_team_user: boolean;
  enabled_ui_pages_internal_users: string[] | null;
  require_auth_for_public_ai_hub: boolean;
  forward_client_headers_to_llm_api: boolean;
  disable_show_blog: boolean;
}

async function fetchUISettings(): Promise<UISettingsData> {
  // getUiSettings returns the raw response: { values: {...}, field_schema: {...} }
  const data = await getUiSettings();
  return data.values as UISettingsData;
}

export function useUISettings() {
  return useQuery<UISettingsData>({
    queryKey: ["uiSettings"],
    queryFn: fetchUISettings,
    staleTime: 5 * 60 * 1000, // 5 minutes
    gcTime: 10 * 60 * 1000,
    retry: 1,
  });
}
