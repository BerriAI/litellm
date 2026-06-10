import { TabPanel, Text, Title } from "@tremor/react";
import PriceDataReload from "@/components/price_data_reload";
import React from "react";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { useModelCostMap } from "../../hooks/models/useModelCostMap";
import { useTranslation } from "react-i18next";

const PriceDataManagementTab = () => {
  const { t } = useTranslation();
  const { accessToken } = useAuthorized();
  const { refetch: refetchModelCostMap } = useModelCostMap();

  return (
    <TabPanel>
      <div className="p-6">
        <div className="mb-6">
          <Title>{t("pages.priceDataManagement.title")}</Title>
          <Text className="text-tremor-content">{t("pages.priceDataManagement.description")}</Text>
        </div>
        <PriceDataReload
          accessToken={accessToken}
          onReloadSuccess={() => {
            refetchModelCostMap();
          }}
          buttonText={t("pages.priceDataManagement.reloadButtonText")}
          size="middle"
          type="primary"
          className="w-full"
        />
      </div>
    </TabPanel>
  );
};

export default PriceDataManagementTab;
