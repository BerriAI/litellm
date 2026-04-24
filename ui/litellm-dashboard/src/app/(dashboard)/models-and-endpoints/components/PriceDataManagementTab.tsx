import PriceDataReload from "@/components/price_data_reload";
import React from "react";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { useModelCostMap } from "../../hooks/models/useModelCostMap";

const PriceDataManagementTab = () => {
  const { accessToken } = useAuthorized();
  const { refetch: refetchModelCostMap } = useModelCostMap();

  return (
    <>
      <div className="p-6">
        <div className="mb-6">
          <h2 className="text-lg font-semibold text-foreground">
            Price Data Management
          </h2>
          <p className="text-muted-foreground">
            Manage model pricing data and configure automatic reload schedules
          </p>
        </div>
        <PriceDataReload
          accessToken={accessToken}
          onReloadSuccess={() => {
            refetchModelCostMap();
          }}
          buttonText="Reload Price Data"
          size="middle"
          type="primary"
          className="w-full"
        />
      </div>
    </>
  );
};

export default PriceDataManagementTab;
