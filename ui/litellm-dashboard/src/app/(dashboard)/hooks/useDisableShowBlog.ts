import { useUISettings } from "./useUISettings";

export function useDisableShowBlog(): boolean {
  const { data } = useUISettings();
  return data?.disable_show_blog ?? false;
}
