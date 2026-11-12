import { useState, type ComponentProps } from "react";
import { useTranslation } from "react-i18next";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import CustomerFallbackFields from "./CustomerFallbackFields";
import CustomerRateLimitFields from "./CustomerRateLimitFields";
import {
  hasIncompleteCustomerFallbacks,
  hasInvalidCustomerRateLimits,
  type CustomerFallbackFields as CustomerFallbackRuleFields,
} from "./customerForms";

export interface CustomerRulesFieldsProps extends ComponentProps<typeof CustomerRateLimitFields> {
  fallbacks: CustomerFallbackRuleFields[] | null;
  onFallbacksChange: (value: CustomerFallbackRuleFields[] | null) => void;
}

export default function CustomerRulesFields({
  fallbacks,
  onFallbacksChange,
  ...rateLimitFields
}: CustomerRulesFieldsProps) {
  const { t } = useTranslation();
  const [expanded, setExpanded] = useState<string[]>([]);
  const showFallbackHint = hasIncompleteCustomerFallbacks(fallbacks) && !expanded.includes("fallbacks");
  const showRateHint = hasInvalidCustomerRateLimits(rateLimitFields) && !expanded.includes("rate-limits");

  return (
    <Accordion multiple value={expanded} onValueChange={setExpanded} className="border-t">
      <AccordionItem value="rate-limits">
        <AccordionTrigger className="py-3">{t("customerPortal.rules.rateLimits")}</AccordionTrigger>
        {showRateHint && (
          <p role="status" className="pb-3 text-xs text-destructive">
            {t("customerPortal.rateLimits.invalidNumber")}
          </p>
        )}
        <AccordionContent className="[&_p:not(:last-child)]:mb-0">
          <CustomerRateLimitFields {...rateLimitFields} />
        </AccordionContent>
      </AccordionItem>
      <AccordionItem value="fallbacks">
        <AccordionTrigger className="py-3">{t("customerPortal.rules.fallbacks")}</AccordionTrigger>
        {showFallbackHint && (
          <p role="status" className="pb-3 text-xs text-destructive">
            {t("customerPortal.fallbacks.selectBackup")}
          </p>
        )}
        <AccordionContent className="[&_p:not(:last-child)]:mb-0">
          <CustomerFallbackFields fallbacks={fallbacks} onChange={onFallbacksChange} />
        </AccordionContent>
      </AccordionItem>
    </Accordion>
  );
}
