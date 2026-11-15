import React, { useState, useEffect } from "react";
import { useTranslation } from "react-i18next";
import { DatabaseIcon } from "@heroicons/react/outline";
import { Badge } from "@/components/ui/badge";
import { vectorStoreListCall } from "../networking";
import { InheritedGrant, inheritedGrantTooltip } from "./inheritedGrants";

interface VectorStoreDetails {
  vector_store_id: string;
  vector_store_name?: string;
}

interface VectorStorePermissionsProps {
  vectorStores: string[];
  inheritedVectorStores?: InheritedGrant[];
  accessToken?: string | null;
}

const STORE_CHIP =
  "inline-flex min-w-0 items-center px-3 py-1.5 rounded-lg bg-info/10 border border-info/20 text-info text-sm font-medium break-words";

export function VectorStorePermissions({
  vectorStores,
  inheritedVectorStores = [],
  accessToken,
}: VectorStorePermissionsProps) {
  const { t } = useTranslation();
  const [vectorStoreDetails, setVectorStoreDetails] = useState<VectorStoreDetails[]>([]);
  const inheritedOnlyStores = inheritedVectorStores.filter((grant) => !vectorStores.includes(grant.id));
  const storeCount = vectorStores.length + inheritedOnlyStores.length;

  // Fetch vector store details when component mounts
  useEffect(() => {
    const fetchVectorStores = async () => {
      if (!accessToken || storeCount === 0) return;

      try {
        const response = await vectorStoreListCall(accessToken);
        if (response.data) {
          setVectorStoreDetails(
            response.data.map((store: any) => ({
              vector_store_id: store.vector_store_id,
              vector_store_name: store.vector_store_name,
            })),
          );
        }
      } catch (error) {
        console.error("Error fetching vector stores:", error);
      }
    };

    fetchVectorStores();
  }, [accessToken, storeCount]);

  // Function to get display name for vector store
  const getVectorStoreDisplayName = (storeId: string) => {
    const storeDetail = vectorStoreDetails.find((store) => store.vector_store_id === storeId);
    if (storeDetail) {
      return `${storeDetail.vector_store_name || storeDetail.vector_store_id} (${storeDetail.vector_store_id})`;
    }
    return storeId;
  };

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2">
        <DatabaseIcon className="h-4 w-4 text-info" />
        <p className="text-sm font-semibold text-foreground">{t("permissions.vectorStorePermissions.title")}</p>
        <Badge variant="secondary">{storeCount}</Badge>
      </div>

      {storeCount > 0 ? (
        <div className="flex flex-wrap gap-2">
          {vectorStores.map((store, index) => (
            <div key={index} className={STORE_CHIP}>
              {getVectorStoreDisplayName(store)}
            </div>
          ))}
          {inheritedOnlyStores.map((grant) => (
            <div key={grant.id} title={inheritedGrantTooltip(grant, t)} className={STORE_CHIP}>
              {getVectorStoreDisplayName(grant.id)}
              {t("permissions.vectorStorePermissions.accessGroupSuffix")}
            </div>
          ))}
        </div>
      ) : (
        <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-muted border border-border">
          <DatabaseIcon className="h-4 w-4 text-muted-foreground" />
          <p className="text-muted-foreground text-sm">{t("permissions.vectorStorePermissions.empty")}</p>
        </div>
      )}
    </div>
  );
}

export default VectorStorePermissions;
