import PriceDataReload from "@/components/price_data_reload";
import React from "react";
import { useTranslation } from "react-i18next";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { useQueryClient } from "@tanstack/react-query";
import { modelCostMapKeys } from "../../hooks/models/useModelCostMap";

const PriceDataManagementTab = () => {
  const { t } = useTranslation();
  const { accessToken } = useAuthorized();
  const queryClient = useQueryClient();

  return (
    <div>
      <div className="p-6">
        <div className="mb-6">
          <h2 className="text-lg font-semibold">{t("modelsAndEndpoints.priceDataManagementTab.title")}</h2>
          <p className="text-sm text-muted-foreground">{t("modelsAndEndpoints.priceDataManagementTab.description")}</p>
        </div>
        <PriceDataReload
          accessToken={accessToken}
          onReloadSuccess={() => {
            queryClient.invalidateQueries({ queryKey: modelCostMapKeys.all });
          }}
          buttonText={t("modelsAndEndpoints.priceDataManagementTab.reloadButton")}
          size="middle"
          type="primary"
          className="w-full"
        />
      </div>
    </div>
  );
};

export default PriceDataManagementTab;
