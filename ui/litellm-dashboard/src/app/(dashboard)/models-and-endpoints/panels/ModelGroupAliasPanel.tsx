"use client";

import { useEffect, useState } from "react";
import ModelGroupAliasSettings from "@/components/model_group_alias_settings";
import { getCallbacksCall, getRouterSettingsCall } from "@/components/networking";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

export default function ModelGroupAliasPanel() {
  const { accessToken, userId: userID, userRole } = useAuthorized();
  const [modelGroupAlias, setModelGroupAlias] = useState<{ [key: string]: string }>({});
  const [aliasOwnership, setAliasOwnership] = useState<{
    accessToken: string;
    userID: typeof userID;
    userRole: typeof userRole;
    managedByConfig: boolean;
  } | null>(null);
  const currentAliasOwnership =
    aliasOwnership?.accessToken === accessToken &&
    aliasOwnership?.userID === userID &&
    aliasOwnership?.userRole === userRole
      ? aliasOwnership
      : null;
  const managedByConfig = currentAliasOwnership?.managedByConfig ?? false;
  const ownershipLoading = Boolean(accessToken && !currentAliasOwnership);

  useEffect(() => {
    if (!accessToken) {
      return;
    }
    let active = true;
    const updateAliasOwnership = (managedByConfig: boolean) => {
      const ownership = { accessToken, userID, userRole, managedByConfig };
      setAliasOwnership(ownership);
    };
    if (userID && userRole) {
      void getCallbacksCall(accessToken, userID, userRole)
        .then((info) => {
          if (active) {
            setModelGroupAlias(info.router_settings?.model_group_alias || {});
          }
        })
        .catch((error) => {
          console.error("Error fetching model group alias:", error);
        });
    }
    void getRouterSettingsCall(accessToken)
      .then((data) => {
        if (active) {
          updateAliasOwnership(data?.source?.model_group_alias === "config");
        }
      })
      .catch((error) => {
        if (active) {
          updateAliasOwnership(false);
        }
        console.error("Error fetching router settings:", error);
      });
    return () => {
      active = false;
    };
  }, [accessToken, userID, userRole]);

  return (
    <ModelGroupAliasSettings
      accessToken={accessToken}
      initialModelGroupAlias={modelGroupAlias}
      onAliasUpdate={setModelGroupAlias}
      managedByConfig={managedByConfig}
      ownershipLoading={ownershipLoading}
    />
  );
}
