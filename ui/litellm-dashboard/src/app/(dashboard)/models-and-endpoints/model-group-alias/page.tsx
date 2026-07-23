"use client";

import { useEffect, useState } from "react";
import ModelGroupAliasSettings from "@/components/model_group_alias_settings";
import { getCallbacksCall } from "@/components/networking";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

export default function ModelGroupAliasPage() {
  const { accessToken, userId: userID, userRole } = useAuthorized();
  const [modelGroupAlias, setModelGroupAlias] = useState<{ [key: string]: string }>({});

  useEffect(() => {
    if (!accessToken || !userID || !userRole) {
      return;
    }
    let active = true;
    void (async () => {
      try {
        const info = await getCallbacksCall(accessToken, userID, userRole);
        if (active) {
          setModelGroupAlias(info.router_settings?.model_group_alias || {});
        }
      } catch (error) {
        console.error("Error fetching model group alias:", error);
      }
    })();
    return () => {
      active = false;
    };
  }, [accessToken, userID, userRole]);

  return (
    <ModelGroupAliasSettings
      accessToken={accessToken}
      initialModelGroupAlias={modelGroupAlias}
      onAliasUpdate={setModelGroupAlias}
    />
  );
}
