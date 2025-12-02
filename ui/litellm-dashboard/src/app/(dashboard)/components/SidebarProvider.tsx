import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import Sidebar from "@/components/leftnav";

interface SidebarProviderProps {
  setPage: (page: string) => void;
  defaultSelectedKey: string;
  sidebarCollapsed: boolean;
}

const SidebarProvider = ({ setPage, defaultSelectedKey, sidebarCollapsed }: SidebarProviderProps) => {
  const { accessToken, userRole } = useAuthorized();

  return (
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
