import React, { useState, useEffect } from "react";
import type { ParseKeys } from "i18next";
import { useTranslation } from "react-i18next";
import { ChevronDown } from "lucide-react";
import {
  AlertDialog,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { Button } from "@/components/ui/button";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { Switch } from "@/components/ui/switch";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { CostTrackingSettingsProps } from "./types";
import ProviderDiscountTable from "./provider_discount_table";
import AddProviderForm from "./add_provider_form";
import ProviderMarginTable from "./provider_margin_table";
import AddMarginForm from "./add_margin_form";
import PricingCalculator from "./pricing_calculator/index";
import { DocsMenu } from "@/components/DocsMenu";
import HowItWorks from "./how_it_works";
import { useDiscountConfig } from "./use_discount_config";
import { useMarginConfig } from "./use_margin_config";
import { useBlockUnpricedConfig } from "./use_block_unpriced_config";
import { fetchAvailableModels, ModelGroup } from "@/components/llm_calls/fetch_models";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";

const DOCS_LINKS: readonly { labelKey: ParseKeys; href: string }[] = [
  {
    labelKey: "costTracking.settings.docsLinks.customPricing",
    href: "https://docs.litellm.ai/docs/proxy/custom_pricing",
  },
  {
    labelKey: "costTracking.settings.docsLinks.spendTracking",
    href: "https://docs.litellm.ai/docs/proxy/cost_tracking",
  },
];

const REMOVAL_COPY = {
  discount: {
    title: "costTracking.settings.removal.discount.title",
    noun: "costTracking.settings.removal.discount.noun",
  },
  margin: { title: "costTracking.settings.removal.margin.title", noun: "costTracking.settings.removal.margin.noun" },
} as const;

interface PendingRemoval {
  kind: keyof typeof REMOVAL_COPY;
  provider: string;
  displayName: string;
}

const SECTION_HEADER_CLASS = "group/section flex w-full items-center justify-between px-6 py-4 text-left";

const SectionHeader: React.FC<{ title: string; description: string }> = ({ title, description }) => (
  <CollapsibleTrigger className={SECTION_HEADER_CLASS}>
    <div className="flex flex-col items-start w-full">
      <span className="block text-lg font-semibold text-foreground">{title}</span>
      <span className="block text-sm text-muted-foreground mt-1">{description}</span>
    </div>
    <ChevronDown className="size-5 shrink-0 text-muted-foreground transition-transform group-data-[panel-open]/section:rotate-180" />
  </CollapsibleTrigger>
);

const CostTrackingSettings: React.FC<CostTrackingSettingsProps> = ({ userID, userRole, accessToken }) => {
  const { t } = useTranslation();
  const [selectedProvider, setSelectedProvider] = useState<string | undefined>(undefined);
  const [newDiscount, setNewDiscount] = useState<string>("");
  const [modelPattern, setModelPattern] = useState<string>("");
  const [isFetching, setIsFetching] = useState(true);
  const [isModalVisible, setIsModalVisible] = useState(false);
  const [isMarginModalVisible, setIsMarginModalVisible] = useState(false);
  const [selectedMarginProvider, setSelectedMarginProvider] = useState<string | undefined>(undefined);
  const [marginType, setMarginType] = useState<"percentage" | "fixed">("percentage");
  const [percentageValue, setPercentageValue] = useState<string>("");
  const [fixedAmountValue, setFixedAmountValue] = useState<string>("");
  const [models, setModels] = useState<string[]>([]);
  const [pendingRemoval, setPendingRemoval] = useState<PendingRemoval | null>(null);
  const [isRemoving, setIsRemoving] = useState(false);

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

  const {
    blockUnpriced,
    isUpdating: isUpdatingBlockUnpriced,
    fetchBlockUnpriced,
    setBlockUnpriced,
  } = useBlockUnpricedConfig({ accessToken });

  useEffect(() => {
    if (accessToken) {
      Promise.all([fetchDiscountConfig(), fetchMarginConfig(), fetchBlockUnpriced()]).finally(() => {
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
  }, [accessToken, fetchDiscountConfig, fetchMarginConfig, fetchBlockUnpriced]);

  const handleAddProvider = async () => {
    const success = await addProvider(selectedProvider, newDiscount, modelPattern);
    if (success) {
      setSelectedProvider(undefined);
      setNewDiscount("");
      setModelPattern("");
      setIsModalVisible(false);
    }
  };

  const handleModalCancel = () => {
    setIsModalVisible(false);
    setSelectedProvider(undefined);
    setNewDiscount("");
    setModelPattern("");
  };

  const handleRemoveProvider = (provider: string, providerDisplayName: string) => {
    setPendingRemoval({ kind: "discount", provider, displayName: providerDisplayName });
  };

  const handleConfirmRemoval = async () => {
    if (!pendingRemoval) return;
    setIsRemoving(true);
    try {
      if (pendingRemoval.kind === "discount") {
        await removeProvider(pendingRemoval.provider);
      } else {
        await removeMargin(pendingRemoval.provider);
      }
    } finally {
      setIsRemoving(false);
      setPendingRemoval(null);
    }
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
    setSelectedMarginProvider(undefined);
    setPercentageValue("");
    setFixedAmountValue("");
    setMarginType("percentage");
  };

  const handleRemoveMargin = (provider: string, providerDisplayName: string) => {
    setPendingRemoval({ kind: "margin", provider, displayName: providerDisplayName });
  };

  if (!accessToken) {
    return null;
  }

  return (
    <div className="w-full p-8">
      {/* Header Section - Outside the card */}
      <div className="flex flex-col md:flex-row items-start md:items-center justify-between mb-6">
        <div>
          <div className="flex items-center gap-2">
            <p className="text-xl font-medium text-foreground">{t("costTracking.settings.title")}</p>
            <DocsMenu items={DOCS_LINKS.map(({ labelKey, href }) => ({ label: t(labelKey), href }))} />
          </div>
          <p className="text-muted-foreground mt-1">{t("costTracking.settings.description")}</p>
        </div>
      </div>

      {/* Main Content Card with Accordions */}
      <div className="bg-card rounded-lg shadow-sm w-full max-w-full space-y-4">
        {/* Accordion 1: Provider Discounts - Only for proxy admins */}
        {isProxyAdmin && (
          <Collapsible className="rounded-lg border">
            <SectionHeader
              title={t("costTracking.settings.discounts.title")}
              description={t("costTracking.settings.discounts.description")}
            />
            <CollapsibleContent className="px-0">
              <Tabs defaultValue="discounts">
                <TabsList variant="line" className="mx-6 mt-4 h-auto justify-start rounded-none border-b p-0">
                  <TabsTrigger value="discounts" className="flex-none rounded-none px-4 py-2">
                    {t("costTracking.settings.discounts.tabs.discounts")}
                  </TabsTrigger>
                  <TabsTrigger value="test-it" className="flex-none rounded-none px-4 py-2">
                    {t("costTracking.settings.discounts.tabs.testIt")}
                  </TabsTrigger>
                </TabsList>
                <TabsContent value="discounts" keepMounted>
                  <div className="p-6">
                    <div className="flex justify-end mb-4">
                      <Button onClick={() => setIsModalVisible(true)}>
                        {t("costTracking.settings.discounts.addButton")}
                      </Button>
                    </div>
                    {isFetching ? (
                      <div className="py-12 text-center">
                        <p className="text-muted-foreground">{t("costTracking.settings.loading")}</p>
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
                          className="mx-auto h-12 w-12 text-muted-foreground mb-4"
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
                        <p className="text-foreground font-medium mb-2">
                          {t("costTracking.settings.discounts.emptyTitle")}
                        </p>
                        <p className="text-muted-foreground text-sm">
                          {t("costTracking.settings.discounts.emptyHint")}
                        </p>
                      </div>
                    )}
                  </div>
                </TabsContent>
                <TabsContent value="test-it" keepMounted>
                  <div className="px-6 pb-4">
                    <HowItWorks />
                  </div>
                </TabsContent>
              </Tabs>
            </CollapsibleContent>
          </Collapsible>
        )}

        {/* Accordion 2: Fee/Price Margin - Only for proxy admins */}
        {isProxyAdmin && (
          <Collapsible className="rounded-lg border">
            <SectionHeader
              title={t("costTracking.settings.margins.title")}
              description={t("costTracking.settings.margins.description")}
            />
            <CollapsibleContent className="px-0">
              <div className="p-6">
                <div className="flex justify-end mb-4">
                  <Button onClick={() => setIsMarginModalVisible(true)}>
                    {t("costTracking.settings.margins.addButton")}
                  </Button>
                </div>
                {isFetching ? (
                  <div className="py-12 text-center">
                    <p className="text-muted-foreground">{t("costTracking.settings.loading")}</p>
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
                      className="mx-auto h-12 w-12 text-muted-foreground mb-4"
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
                    <p className="text-foreground font-medium mb-2">{t("costTracking.settings.margins.emptyTitle")}</p>
                    <p className="text-muted-foreground text-sm">{t("costTracking.settings.margins.emptyHint")}</p>
                  </div>
                )}
              </div>
            </CollapsibleContent>
          </Collapsible>
        )}

        {/* Accordion 3: Block Unpriced Models - Only for proxy admins */}
        {isProxyAdmin && (
          <Collapsible className="rounded-lg border">
            <SectionHeader
              title={t("costTracking.settings.blockUnpriced.title")}
              description={t("costTracking.settings.blockUnpriced.description")}
            />
            <CollapsibleContent className="px-0">
              <div className="p-6">
                <div className="flex items-center justify-between">
                  <div className="pr-6">
                    <p className="text-foreground font-medium">
                      {t("costTracking.settings.blockUnpriced.toggleTitle")}
                    </p>
                    <p className="text-sm text-muted-foreground mt-1">
                      {t("costTracking.settings.blockUnpriced.toggleDescription")}
                    </p>
                  </div>
                  <Switch
                    checked={blockUnpriced}
                    disabled={isUpdatingBlockUnpriced || isFetching}
                    onCheckedChange={(checked) => setBlockUnpriced(checked)}
                  />
                </div>
              </div>
            </CollapsibleContent>
          </Collapsible>
        )}

        {/* Accordion 4: Pricing Calculator - Available to all roles */}
        <Collapsible defaultOpen={true} className="rounded-lg border">
          <SectionHeader
            title={t("costTracking.settings.pricingCalculator.title")}
            description={t("costTracking.settings.pricingCalculator.description")}
          />
          <CollapsibleContent className="px-0">
            <div className="p-6">
              <PricingCalculator accessToken={accessToken} models={models} />
            </div>
          </CollapsibleContent>
        </Collapsible>
      </div>

      {pendingRemoval && (
        <AlertDialog open onOpenChange={(open) => !open && !isRemoving && setPendingRemoval(null)}>
          <AlertDialogContent>
            <AlertDialogHeader>
              <AlertDialogTitle>{t(REMOVAL_COPY[pendingRemoval.kind].title)}</AlertDialogTitle>
              <AlertDialogDescription>
                {t("costTracking.settings.removal.confirm", {
                  noun: t(REMOVAL_COPY[pendingRemoval.kind].noun),
                  name: pendingRemoval.displayName,
                })}
              </AlertDialogDescription>
            </AlertDialogHeader>
            <AlertDialogFooter>
              <AlertDialogCancel disabled={isRemoving}>{t("costTracking.settings.removal.cancel")}</AlertDialogCancel>
              <Button variant="destructive" onClick={handleConfirmRemoval} disabled={isRemoving}>
                {isRemoving ? t("costTracking.settings.removal.removing") : t("costTracking.settings.removal.remove")}
              </Button>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialog>
      )}

      <Dialog open={isModalVisible} onOpenChange={(open) => !open && handleModalCancel()}>
        <DialogContent className="top-8 max-h-[calc(100dvh-4rem)] translate-y-0 overflow-y-auto sm:max-w-[1000px]">
          <DialogHeader>
            <div className="flex items-center space-x-3 pb-4 border-b border-border">
              <DialogTitle className="text-xl font-semibold text-foreground">
                {t("costTracking.settings.discounts.dialogTitle")}
              </DialogTitle>
            </div>
          </DialogHeader>
          <div className="mt-6">
            <p className="text-sm text-muted-foreground mb-6">
              {t("costTracking.settings.discounts.dialogDescription")}
            </p>
            <form onSubmit={(event) => event.preventDefault()} className="space-y-6">
              <AddProviderForm
                selectedProvider={selectedProvider}
                newDiscount={newDiscount}
                modelPattern={modelPattern}
                onProviderChange={setSelectedProvider}
                onDiscountChange={setNewDiscount}
                onModelPatternChange={setModelPattern}
                onAddProvider={handleAddProvider}
              />
            </form>
          </div>
        </DialogContent>
      </Dialog>

      <Dialog open={isMarginModalVisible} onOpenChange={(open) => !open && handleMarginModalCancel()}>
        <DialogContent className="top-8 max-h-[calc(100dvh-4rem)] translate-y-0 overflow-y-auto sm:max-w-[1000px]">
          <DialogHeader>
            <div className="flex items-center space-x-3 pb-4 border-b border-border">
              <DialogTitle className="text-xl font-semibold text-foreground">
                {t("costTracking.settings.margins.dialogTitle")}
              </DialogTitle>
            </div>
          </DialogHeader>
          <div className="mt-6">
            <p className="text-sm text-muted-foreground mb-6">{t("costTracking.settings.margins.dialogDescription")}</p>
            <form onSubmit={(event) => event.preventDefault()} className="space-y-6">
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
            </form>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
};

export default CostTrackingSettings;
