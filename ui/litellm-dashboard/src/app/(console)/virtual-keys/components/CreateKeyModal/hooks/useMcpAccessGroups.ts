import { useEffect, useState } from "react";
import { getMCPAccessGroups } from "@/app/(console)/virtual-keys/components/CreateKeyModal/networking";
import useAuthorized from "@/app/(console)/hooks/useAuthorized";

export const useMcpAccessGroups = () => {
  const [mcpAccessGroups, setMcpAccessGroups] = useState<string[]>([]);
  const { accessToken } = useAuthorized();

  useEffect(() => {
    if (!accessToken) {
      setMcpAccessGroups([]);
      return;
    }
    (async () => {
      const groups = await getMCPAccessGroups(accessToken);
      setMcpAccessGroups(groups || []);
    })();
  }, [accessToken]);

  return mcpAccessGroups;
};
