import { useUISettingsFlags } from "./useUISettingsFlags";

export function useDisableShowBlog(): boolean {
  const { data } = useUISettingsFlags();
  return data?.disable_show_blog ?? false;
}
