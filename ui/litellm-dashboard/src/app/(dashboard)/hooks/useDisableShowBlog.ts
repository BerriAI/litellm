import { useUISettings } from "./uiSettings/useUISettings";

export function useDisableShowBlog(): boolean {
  const { data } = useUISettings();
  return (data?.values?.disable_show_blog as boolean) ?? false;
}
