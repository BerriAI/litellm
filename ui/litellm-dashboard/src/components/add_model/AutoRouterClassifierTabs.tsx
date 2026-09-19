import React, { useId } from "react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { effectiveClassifierType, type ComplexityRouterConfigValue } from "./ComplexityRouterConfig";
import { transitionClassifierType } from "./classifier_type_transition";
import { isForecastClassifier } from "./forecast_classifier_config";

interface AutoRouterClassifierTabsProps {
  value: ComplexityRouterConfigValue;
  onChange: (value: ComplexityRouterConfigValue) => void;
  children: React.ReactNode;
}

const AutoRouterClassifierTabs: React.FC<AutoRouterClassifierTabsProps> = ({ value, onChange, children }) => {
  const restrictionId = useId();
  const classifierType = effectiveClassifierType(value);
  const selected = isForecastClassifier(classifierType) ? classifierType : "complexity";
  const hasCustomTiers = Boolean(value.custom_tier_set);

  const handleChange = (tab: unknown) => {
    if (tab === selected) return;
    if (tab === "complexity") {
      onChange(transitionClassifierType(value, isForecastClassifier(classifierType) ? "heuristic" : classifierType));
    } else if (!hasCustomTiers && (tab === "capability" || tab === "llm_v2")) {
      onChange(transitionClassifierType(value, tab));
    }
  };

  return (
    <Tabs value={selected} onValueChange={handleChange}>
      <p className="text-sm font-medium">Classifier type</p>
      <TabsList aria-label="Classifier type" className="w-full">
        <TabsTrigger value="complexity">Complexity</TabsTrigger>
        <TabsTrigger
          value="capability"
          disabled={hasCustomTiers}
          aria-describedby={hasCustomTiers ? restrictionId : undefined}
        >
          Capability
        </TabsTrigger>
        <TabsTrigger
          value="llm_v2"
          disabled={hasCustomTiers}
          aria-describedby={hasCustomTiers ? restrictionId : undefined}
        >
          Fuse v2
        </TabsTrigger>
      </TabsList>
      {hasCustomTiers && (
        <p id={restrictionId} className="text-sm text-muted-foreground">
          Restore standard tiers to use Capability or Fuse v2.
        </p>
      )}
      <TabsContent value={selected}>{children}</TabsContent>
    </Tabs>
  );
};

export default AutoRouterClassifierTabs;
