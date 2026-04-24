import React, { useState, useEffect } from "react";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { Button, buttonVariants } from "@/components/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { cn } from "@/lib/utils";
import { CostTrackingSettingsProps } from "./types";
import ProviderDiscountTable from "./provider_discount_table";
import AddProviderForm from "./add_provider_form";
import ProviderMarginTable from "./provider_margin_table";
import AddMarginForm from "./add_margin_form";
import PricingCalculator from "./pricing_calculator/index";
import { AlertCircle } from "lucide-react";
import { DocsMenu } from "../HelpLink";
import HowItWorks from "./how_it_works";
import { useDiscountConfig } from "./use_discount_config";
import { useMarginConfig } from "./use_margin_config";
import { fetchAvailableModels, ModelGroup } from "../playground/llm_calls/fetch_models";

const DOCS_LINKS = [
  { label: "Custom pricing for models", href: "https://docs.litellm.ai/docs/proxy/custom_pricing" },
  { label: "Spend tracking", href: "https://docs.litellm.ai/docs/proxy/cost_tracking" },
];

interface RemoveConfirmState {
  open: boolean;
  variant: "discount" | "margin";
  provider: string;
  providerDisplayName: string;
}

const DEFAULT_REMOVE_CONFIRM: RemoveConfirmState = {
  open: false,
  variant: "discount",
  provider: "",
  providerDisplayName: "",
};

const EmptyStateIllustration: React.FC = () => (
  <svg className="mx-auto h-12 w-12 text-muted-foreground mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path
      strokeLinecap="round"
      strokeLinejoin="round"
      strokeWidth={1.5}
      d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
    />
  </svg>
);

const CostTrackingSettings: React.FC<CostTrackingSettingsProps> = ({ userID, userRole, accessToken }) => {
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
  const [removeConfirm, setRemoveConfirm] = useState<RemoveConfirmState>(DEFAULT_REMOVE_CONFIRM);

  const isProxyAdmin = userRole === "proxy_admin" || userRole === "Admin";

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

  const handleDiscountModalOpenChange = (open: boolean) => {
    setIsModalVisible(open);
    if (!open) {
      setSelectedProvider(undefined);
      setNewDiscount("");
    }
  };

  const handleRemoveProvider = (provider: string, providerDisplayName: string) => {
    setRemoveConfirm({
      open: true,
      variant: "discount",
      provider,
      providerDisplayName,
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

  const handleMarginModalOpenChange = (open: boolean) => {
    setIsMarginModalVisible(open);
    if (!open) {
      setSelectedMarginProvider(undefined);
      setPercentageValue("");
      setFixedAmountValue("");
      setMarginType("percentage");
    }
  };

  const handleRemoveMargin = (provider: string, providerDisplayName: string) => {
    setRemoveConfirm({
      open: true,
      variant: "margin",
      provider,
      providerDisplayName,
    });
  };

  const handleConfirmRemove = async () => {
    if (removeConfirm.variant === "discount") {
      await removeProvider(removeConfirm.provider);
    } else {
      await removeMargin(removeConfirm.provider);
    }
    setRemoveConfirm(DEFAULT_REMOVE_CONFIRM);
  };

  const handleRemoveDialogOpenChange = (open: boolean) => {
    if (!open) {
      setRemoveConfirm(DEFAULT_REMOVE_CONFIRM);
    }
  };

  if (!accessToken) {
    return null;
  }

  const defaultAccordionValue: string[] = isProxyAdmin
    ? ["provider-discounts", "fee-price-margin", "pricing-calculator"]
    : ["pricing-calculator"];

  return (
    <div className="w-full p-8">
      <div className="flex flex-col md:flex-row items-start md:items-center justify-between mb-6">
        <div>
          <div className="flex items-center gap-2">
            <h2 className="text-2xl font-semibold m-0">Cost Tracking Settings</h2>
            <DocsMenu items={DOCS_LINKS} />
          </div>
          <p className="text-muted-foreground text-sm mt-1">
            Configure cost discounts and margins for different LLM providers. Changes are saved automatically.
          </p>
        </div>
      </div>

      <div className="bg-card rounded-lg shadow w-full max-w-full">
        <Accordion type="multiple" defaultValue={defaultAccordionValue}>
          {isProxyAdmin && (
            <AccordionItem value="provider-discounts" className="border-b">
              <AccordionTrigger className="px-6 py-4 hover:no-underline">
                <div className="flex flex-col items-start w-full">
                  <span className="text-lg font-semibold text-foreground">Provider Discounts</span>
                  <span className="text-sm text-muted-foreground mt-1">
                    Apply percentage-based discounts to reduce costs for specific providers
                  </span>
                </div>
              </AccordionTrigger>
              <AccordionContent className="p-0">
                <Tabs defaultValue="discounts">
                  <TabsList className="mx-6 mt-4">
                    <TabsTrigger value="discounts">Discounts</TabsTrigger>
                    <TabsTrigger value="test-it">Test It</TabsTrigger>
                  </TabsList>
                  <TabsContent value="discounts">
                    <div className="p-6">
                      <div className="flex justify-end mb-4">
                        <Button onClick={() => setIsModalVisible(true)}>+ Add Provider Discount</Button>
                      </div>
                      {isFetching ? (
                        <div className="py-12 text-center">
                          <span className="text-muted-foreground">Loading configuration...</span>
                        </div>
                      ) : Object.keys(discountConfig).length > 0 ? (
                        <ProviderDiscountTable
                          discountConfig={discountConfig}
                          onDiscountChange={handleDiscountChange}
                          onRemoveProvider={handleRemoveProvider}
                        />
                      ) : (
                        <div className="py-16 px-6 text-center">
                          <EmptyStateIllustration />
                          <div className="text-foreground font-medium mb-2">No provider discounts configured</div>
                          <div className="text-muted-foreground text-sm">
                            Click &quot;Add Provider Discount&quot; to get started
                          </div>
                        </div>
                      )}
                    </div>
                  </TabsContent>
                  <TabsContent value="test-it">
                    <div className="px-6 pb-4">
                      <HowItWorks />
                    </div>
                  </TabsContent>
                </Tabs>
              </AccordionContent>
            </AccordionItem>
          )}

          {isProxyAdmin && (
            <AccordionItem value="fee-price-margin" className="border-b">
              <AccordionTrigger className="px-6 py-4 hover:no-underline">
                <div className="flex flex-col items-start w-full">
                  <span className="text-lg font-semibold text-foreground">Fee/Price Margin</span>
                  <span className="text-sm text-muted-foreground mt-1">
                    Add fees or margins to LLM costs for internal billing and cost recovery
                  </span>
                </div>
              </AccordionTrigger>
              <AccordionContent className="p-0">
                <div className="p-6">
                  <div className="flex justify-end mb-4">
                    <Button onClick={() => setIsMarginModalVisible(true)}>+ Add Provider Margin</Button>
                  </div>
                  {isFetching ? (
                    <div className="py-12 text-center">
                      <span className="text-muted-foreground">Loading configuration...</span>
                    </div>
                  ) : Object.keys(marginConfig).length > 0 ? (
                    <ProviderMarginTable
                      marginConfig={marginConfig}
                      onMarginChange={handleMarginChange}
                      onRemoveProvider={handleRemoveMargin}
                    />
                  ) : (
                    <div className="py-16 px-6 text-center">
                      <EmptyStateIllustration />
                      <div className="text-foreground font-medium mb-2">No provider margins configured</div>
                      <div className="text-muted-foreground text-sm">
                        Click &quot;Add Provider Margin&quot; to get started
                      </div>
                    </div>
                  )}
                </div>
              </AccordionContent>
            </AccordionItem>
          )}

          <AccordionItem value="pricing-calculator" className="border-b-0">
            <AccordionTrigger className="px-6 py-4 hover:no-underline">
              <div className="flex flex-col items-start w-full">
                <span className="text-lg font-semibold text-foreground">Pricing Calculator</span>
                <span className="text-sm text-muted-foreground mt-1">
                  Estimate LLM costs based on expected token usage and request volume
                </span>
              </div>
            </AccordionTrigger>
            <AccordionContent className="p-0">
              <div className="p-6">
                <PricingCalculator accessToken={accessToken} models={models} />
              </div>
            </AccordionContent>
          </AccordionItem>
        </Accordion>
      </div>

      {isModalVisible && (
        <Dialog open onOpenChange={handleDiscountModalOpenChange}>
          <DialogContent className="max-w-[1000px] top-8 translate-y-0">
            <DialogHeader className="pb-4 border-b border-border">
              <DialogTitle asChild>
                <h2 className="text-xl font-semibold text-foreground">Add Provider Discount</h2>
              </DialogTitle>
              <DialogDescription className="sr-only">Add a provider discount</DialogDescription>
            </DialogHeader>
            <div className="mt-6">
              <p className="text-sm text-muted-foreground mb-6">
                Select a provider and set its discount percentage. Enter a value between 0% and 100% (e.g., 5 for a 5%
                discount).
              </p>
              <AddProviderForm
                discountConfig={discountConfig}
                selectedProvider={selectedProvider}
                newDiscount={newDiscount}
                onProviderChange={setSelectedProvider}
                onDiscountChange={setNewDiscount}
                onAddProvider={handleAddProvider}
              />
            </div>
          </DialogContent>
        </Dialog>
      )}

      {isMarginModalVisible && (
        <Dialog open onOpenChange={handleMarginModalOpenChange}>
          <DialogContent className="max-w-[1000px] top-8 translate-y-0">
            <DialogHeader className="pb-4 border-b border-border">
              <DialogTitle asChild>
                <h2 className="text-xl font-semibold text-foreground">Add Provider Margin</h2>
              </DialogTitle>
              <DialogDescription className="sr-only">Add a provider margin</DialogDescription>
            </DialogHeader>
            <div className="mt-6">
              <p className="text-sm text-muted-foreground mb-6">
                Select a provider (or &quot;Global&quot; for all providers) and configure the margin. You can use
                percentage-based or fixed amount.
              </p>
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
            </div>
          </DialogContent>
        </Dialog>
      )}

      {removeConfirm.open && (
        <AlertDialog open onOpenChange={handleRemoveDialogOpenChange}>
          <AlertDialogContent>
            <AlertDialogHeader>
              <AlertDialogTitle className="flex items-center gap-2">
                <AlertCircle className="h-5 w-5 text-destructive" />
                {removeConfirm.variant === "discount" ? "Remove Provider Discount" : "Remove Provider Margin"}
              </AlertDialogTitle>
              <AlertDialogDescription>
                {removeConfirm.variant === "discount"
                  ? `Are you sure you want to remove the discount for ${removeConfirm.providerDisplayName}?`
                  : `Are you sure you want to remove the margin for ${removeConfirm.providerDisplayName}?`}
              </AlertDialogDescription>
            </AlertDialogHeader>
            <AlertDialogFooter>
              <AlertDialogCancel>Cancel</AlertDialogCancel>
              <AlertDialogAction
                onClick={handleConfirmRemove}
                className={cn(buttonVariants({ variant: "destructive" }))}
              >
                Remove
              </AlertDialogAction>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialog>
      )}
    </div>
  );
};

export default CostTrackingSettings;
