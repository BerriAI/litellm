import React, { useState, useEffect, useMemo } from "react";
import {
  Title,
  Text,
  Button,
  Accordion,
  AccordionHeader,
  AccordionBody,
  TabGroup,
  TabList,
  Tab,
  TabPanels,
  TabPanel,
} from "@tremor/react";
import { Modal, Form } from "antd";
import { useTranslation } from "react-i18next";
import { CostTrackingSettingsProps } from "./types";
import ProviderDiscountTable from "./provider_discount_table";
import AddProviderForm from "./add_provider_form";
import ProviderMarginTable from "./provider_margin_table";
import AddMarginForm from "./add_margin_form";
import PricingCalculator from "./pricing_calculator/index";
import { ExclamationCircleOutlined } from "@ant-design/icons";
import { DocsMenu } from "../HelpLink";
import HowItWorks from "./how_it_works";
import { useDiscountConfig } from "./use_discount_config";
import { useMarginConfig } from "./use_margin_config";
import { fetchAvailableModels, ModelGroup } from "@/components/llm_calls/fetch_models";

const CostTrackingSettings: React.FC<CostTrackingSettingsProps> = ({ userID, userRole, accessToken }) => {
  const { t } = useTranslation();
  const docsLinks = useMemo(
    () => [
      {
        label: t("costTracking.costTrackingSettings.docsLinkCustomPricing"),
        href: "https://docs.litellm.ai/docs/proxy/custom_pricing",
      },
      {
        label: t("costTracking.costTrackingSettings.docsLinkSpendTracking"),
        href: "https://docs.litellm.ai/docs/proxy/cost_tracking",
      },
    ],
    [t],
  );
  const [selectedProvider, setSelectedProvider] = useState<string | undefined>(undefined);
  const [newDiscount, setNewDiscount] = useState<string>("");
  const [isFetching, setIsFetching] = useState(true);
  const [isModalVisible, setIsModalVisible] = useState(false);
  const [isMarginModalVisible, setIsMarginModalVisible] = useState(false);
  const [selectedMarginProvider, setSelectedMarginProvider] = useState<string | undefined>(undefined);
  const [marginType, setMarginType] = useState<"percentage" | "fixed">("percentage");
  const [percentageValue, setPercentageValue] = useState<string>("");
  const [fixedAmountValue, setFixedAmountValue] = useState<string>("");
  const [models, setModels] = useState<string[]>([]);
  const [form] = Form.useForm();
  const [marginForm] = Form.useForm();
  const [modal, contextHolder] = Modal.useModal();

  const isProxyAdmin = userRole === "proxy_admin" || userRole === "Admin";

  // Use custom hooks for discount and margin config
  const {
    discountConfig,
    fetchDiscountConfig,
    handleAddProvider: addProvider,
    handleRemoveProvider: removeProvider,
    handleDiscountChange,
  } = useDiscountConfig({ accessToken });

  const {
    marginConfig,
    fetchMarginConfig,
    handleAddMargin: addMargin,
    handleRemoveMargin: removeMargin,
    handleMarginChange,
  } = useMarginConfig({ accessToken });

  useEffect(() => {
    if (accessToken) {
      Promise.all([fetchDiscountConfig(), fetchMarginConfig()]).finally(() => {
        setIsFetching(false);
      });

      // Fetch models for pricing calculator (available to all roles)
      const loadModels = async () => {
        try {
          const modelGroups = await fetchAvailableModels(accessToken);
          setModels(modelGroups.map((m: ModelGroup) => m.model_group));
        } catch (error) {
          console.error("Error fetching models:", error);
        }
      };
      loadModels();
    }
  }, [accessToken, fetchDiscountConfig, fetchMarginConfig]);

  const handleAddProvider = async () => {
    const success = await addProvider(selectedProvider, newDiscount);
    if (success) {
      setSelectedProvider(undefined);
      setNewDiscount("");
      setIsModalVisible(false);
    }
  };

  const handleModalCancel = () => {
    setIsModalVisible(false);
    form.resetFields();
    setSelectedProvider(undefined);
    setNewDiscount("");
  };

  const handleFormSubmit = () => {
    handleAddProvider();
  };

  const handleRemoveProvider = async (provider: string, providerDisplayName: string) => {
    modal.confirm({
      title: t("costTracking.costTrackingSettings.removeDiscountTitle"),
      icon: <ExclamationCircleOutlined />,
      content: t("costTracking.costTrackingSettings.removeDiscountContent", { providerDisplayName }),
      okText: t("costTracking.costTrackingSettings.removeOkText"),
      okType: "danger",
      cancelText: t("common.cancel"),
      onOk: () => removeProvider(provider),
    });
  };

  const handleAddMargin = async () => {
    const success = await addMargin({
      selectedProvider: selectedMarginProvider,
      marginType,
      percentageValue,
      fixedAmountValue,
    });
    if (success) {
      setSelectedMarginProvider(undefined);
      setPercentageValue("");
      setFixedAmountValue("");
      setMarginType("percentage");
      setIsMarginModalVisible(false);
    }
  };

  const handleMarginModalCancel = () => {
    setIsMarginModalVisible(false);
    marginForm.resetFields();
    setSelectedMarginProvider(undefined);
    setPercentageValue("");
    setFixedAmountValue("");
    setMarginType("percentage");
  };

  const handleRemoveMargin = async (provider: string, providerDisplayName: string) => {
    modal.confirm({
      title: t("costTracking.costTrackingSettings.removeMarginTitle"),
      icon: <ExclamationCircleOutlined />,
      content: t("costTracking.costTrackingSettings.removeMarginContent", { providerDisplayName }),
      okText: t("costTracking.costTrackingSettings.removeOkText"),
      okType: "danger",
      cancelText: t("common.cancel"),
      onOk: () => removeMargin(provider),
    });
  };

  if (!accessToken) {
    return null;
  }

  return (
    <div className="w-full p-8">
      {contextHolder}

      {/* Header Section - Outside the card */}
      <div className="flex flex-col md:flex-row items-start md:items-center justify-between mb-6">
        <div>
          <div className="flex items-center gap-2">
            <Title>{t("costTracking.costTrackingSettings.pageTitle")}</Title>
            <DocsMenu items={docsLinks} />
          </div>
          <Text className="text-gray-500 mt-1">{t("costTracking.costTrackingSettings.pageSubtitle")}</Text>
        </div>
      </div>

      {/* Main Content Card with Accordions */}
      <div className="bg-white rounded-lg shadow w-full max-w-full space-y-4">
        {/* Accordion 1: Provider Discounts - Only for proxy admins */}
        {isProxyAdmin && (
          <Accordion>
            <AccordionHeader className="px-6 py-4">
              <div className="flex flex-col items-start w-full">
                <Text className="text-lg font-semibold text-gray-900">
                  {t("costTracking.costTrackingSettings.providerDiscountsTitle")}
                </Text>
                <Text className="text-sm text-gray-500 mt-1">
                  {t("costTracking.costTrackingSettings.providerDiscountsSubtitle")}
                </Text>
              </div>
            </AccordionHeader>
            <AccordionBody className="px-0">
              <TabGroup>
                <TabList className="px-6 pt-4">
                  <Tab>{t("costTracking.costTrackingSettings.discountsTab")}</Tab>
                  <Tab>{t("costTracking.costTrackingSettings.testItTab")}</Tab>
                </TabList>
                <TabPanels>
                  <TabPanel>
                    <div className="p-6">
                      <div className="flex justify-end mb-4">
                        <Button onClick={() => setIsModalVisible(true)}>
                          {t("costTracking.costTrackingSettings.addProviderDiscountButton")}
                        </Button>
                      </div>
                      {isFetching ? (
                        <div className="py-12 text-center">
                          <Text className="text-gray-500">{t("common.loading")}</Text>
                        </div>
                      ) : Object.keys(discountConfig).length > 0 ? (
                        <ProviderDiscountTable
                          discountConfig={discountConfig}
                          onDiscountChange={handleDiscountChange}
                          onRemoveProvider={handleRemoveProvider}
                        />
                      ) : (
                        <div className="py-16 px-6 text-center">
                          <svg
                            className="mx-auto h-12 w-12 text-gray-400 mb-4"
                            fill="none"
                            stroke="currentColor"
                            viewBox="0 0 24 24"
                          >
                            <path
                              strokeLinecap="round"
                              strokeLinejoin="round"
                              strokeWidth={1.5}
                              d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                            />
                          </svg>
                          <Text className="text-gray-700 font-medium mb-2">
                            {t("costTracking.costTrackingSettings.noDiscountsConfigured")}
                          </Text>
                          <Text className="text-gray-500 text-sm">
                            {t("costTracking.costTrackingSettings.noDiscountsHint")}
                          </Text>
                        </div>
                      )}
                    </div>
                  </TabPanel>
                  <TabPanel>
                    <div className="px-6 pb-4">
                      <HowItWorks />
                    </div>
                  </TabPanel>
                </TabPanels>
              </TabGroup>
            </AccordionBody>
          </Accordion>
        )}

        {/* Accordion 2: Fee/Price Margin - Only for proxy admins */}
        {isProxyAdmin && (
          <Accordion>
            <AccordionHeader className="px-6 py-4">
              <div className="flex flex-col items-start w-full">
                <Text className="text-lg font-semibold text-gray-900">
                  {t("costTracking.costTrackingSettings.feePriceMarginTitle")}
                </Text>
                <Text className="text-sm text-gray-500 mt-1">
                  {t("costTracking.costTrackingSettings.feePriceMarginSubtitle")}
                </Text>
              </div>
            </AccordionHeader>
            <AccordionBody className="px-0">
              <div className="p-6">
                <div className="flex justify-end mb-4">
                  <Button onClick={() => setIsMarginModalVisible(true)}>
                    {t("costTracking.costTrackingSettings.addProviderMarginButton")}
                  </Button>
                </div>
                {isFetching ? (
                  <div className="py-12 text-center">
                    <Text className="text-gray-500">{t("common.loading")}</Text>
                  </div>
                ) : Object.keys(marginConfig).length > 0 ? (
                  <ProviderMarginTable
                    marginConfig={marginConfig}
                    onMarginChange={handleMarginChange}
                    onRemoveProvider={handleRemoveMargin}
                  />
                ) : (
                  <div className="py-16 px-6 text-center">
                    <svg
                      className="mx-auto h-12 w-12 text-gray-400 mb-4"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={1.5}
                        d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                      />
                    </svg>
                    <Text className="text-gray-700 font-medium mb-2">
                      {t("costTracking.costTrackingSettings.noMarginsConfigured")}
                    </Text>
                    <Text className="text-gray-500 text-sm">
                      {t("costTracking.costTrackingSettings.noMarginsHint")}
                    </Text>
                  </div>
                )}
              </div>
            </AccordionBody>
          </Accordion>
        )}

        {/* Accordion 3: Pricing Calculator - Available to all roles */}
        <Accordion defaultOpen={true}>
          <AccordionHeader className="px-6 py-4">
            <div className="flex flex-col items-start w-full">
              <Text className="text-lg font-semibold text-gray-900">
                {t("costTracking.costTrackingSettings.pricingCalculatorTitle")}
              </Text>
              <Text className="text-sm text-gray-500 mt-1">
                {t("costTracking.costTrackingSettings.pricingCalculatorSubtitle")}
              </Text>
            </div>
          </AccordionHeader>
          <AccordionBody className="px-0">
            <div className="p-6">
              <PricingCalculator accessToken={accessToken} models={models} />
            </div>
          </AccordionBody>
        </Accordion>
      </div>

      <Modal
        title={
          <div className="flex items-center space-x-3 pb-4 border-b border-gray-100">
            <h2 className="text-xl font-semibold text-gray-900">
              {t("costTracking.costTrackingSettings.addDiscountModalTitle")}
            </h2>
          </div>
        }
        open={isModalVisible}
        width={1000}
        onCancel={handleModalCancel}
        footer={null}
        className="top-8"
        styles={{
          body: { padding: "24px" },
          header: { padding: "24px 24px 0 24px", border: "none" },
        }}
      >
        <div className="mt-6">
          <Text className="text-sm text-gray-600 mb-6">
            {t("costTracking.costTrackingSettings.addDiscountModalDesc")}
          </Text>
          <Form form={form} onFinish={handleFormSubmit} layout="vertical" className="space-y-6">
            <AddProviderForm
              discountConfig={discountConfig}
              selectedProvider={selectedProvider}
              newDiscount={newDiscount}
              onProviderChange={setSelectedProvider}
              onDiscountChange={setNewDiscount}
              onAddProvider={handleAddProvider}
            />
          </Form>
        </div>
      </Modal>

      <Modal
        title={
          <div className="flex items-center space-x-3 pb-4 border-b border-gray-100">
            <h2 className="text-xl font-semibold text-gray-900">
              {t("costTracking.costTrackingSettings.addMarginModalTitle")}
            </h2>
          </div>
        }
        open={isMarginModalVisible}
        width={1000}
        onCancel={handleMarginModalCancel}
        footer={null}
        className="top-8"
        styles={{
          body: { padding: "24px" },
          header: { padding: "24px 24px 0 24px", border: "none" },
        }}
      >
        <div className="mt-6">
          <Text className="text-sm text-gray-600 mb-6">
            {t("costTracking.costTrackingSettings.addMarginModalDesc")}
          </Text>
          <Form form={marginForm} layout="vertical" className="space-y-6">
            <AddMarginForm
              marginConfig={marginConfig}
              selectedProvider={selectedMarginProvider}
              marginType={marginType}
              percentageValue={percentageValue}
              fixedAmountValue={fixedAmountValue}
              onProviderChange={setSelectedMarginProvider}
              onMarginTypeChange={setMarginType}
              onPercentageChange={setPercentageValue}
              onFixedAmountChange={setFixedAmountValue}
              onAddProvider={handleAddMargin}
            />
          </Form>
        </div>
      </Modal>
    </div>
  );
};

export default CostTrackingSettings;
