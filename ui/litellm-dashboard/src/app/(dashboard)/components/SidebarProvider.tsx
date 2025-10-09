import useFeatureFlags from "@/hooks/useFeatureFlags";
import Sidebar from "@/components/leftnav";
import Sidebar2 from "@/app/(dashboard)/components/Sidebar2";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

interface SidebarProviderProps {
  defaultSelectedKey: string;
  setPage: (newPage: string) => void;
  sidebarCollapsed: boolean;
}

const SidebarProvider = ({ setPage, defaultSelectedKey, sidebarCollapsed }: SidebarProviderProps) => {
  const { refactoredUIFlag } = useFeatureFlags();
  const { accessToken, userRole } = useAuthorized();

  return refactoredUIFlag ? (
    <Sidebar2 accessToken={accessToken} defaultSelectedKey={defaultSelectedKey} userRole={userRole} />
  ) : (
    <Sidebar
      accessToken={accessToken}
      setPage={setPage}
      userRole={userRole}
      defaultSelectedKey={defaultSelectedKey}
      collapsed={sidebarCollapsed}
    />
  );
};

export default SidebarProvider;
