import { PortalShell } from "@/components/public-relay/PortalContext";

export default function RelayPortalLayout({ children }: { children: React.ReactNode }) {
  return <PortalShell>{children}</PortalShell>;
}
