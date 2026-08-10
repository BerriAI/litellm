import { Text, Title } from "@tremor/react";
import PriceDataReload from "@/components/price_data_reload";
import React from "react";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { useModelCostMap } from "../../hooks/models/useModelCostMap";
import { useTranslation } from "react-i18next";

const PriceDataManagementTab = () => {
  const { t } = useTranslation("gateway");
  const { accessToken } = useAuthorized();
  const { refetch: refetchModelCostMap } = useModelCostMap();

  return (
    <div>
      <div className="p-6">
        <div className="mb-6">
          <Title>{t("models.priceData.title")}</Title>
          <Text className="text-tremor-content">{t("models.priceData.description")}</Text>
        </div>
        <PriceDataReload
          accessToken={accessToken}
          onReloadSuccess={() => {
            refetchModelCostMap();
          }}
          buttonText={t("models.priceData.reload")}
          size="middle"
          type="primary"
          className="w-full"
        />
      </div>
    </div>
  );
};

export default PriceDataManagementTab;
