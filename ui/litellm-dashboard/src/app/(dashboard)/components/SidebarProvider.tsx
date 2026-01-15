import Sidebar from "@/components/leftnav";

interface SidebarProviderProps {
  setPage: (page: string) => void;
  defaultSelectedKey: string;
  sidebarCollapsed: boolean;
}

const SidebarProvider = ({ setPage, defaultSelectedKey, sidebarCollapsed }: SidebarProviderProps) => {
  return <Sidebar setPage={setPage} defaultSelectedKey={defaultSelectedKey} collapsed={sidebarCollapsed} />;
};

export default SidebarProvider;
