import React, { useState, useMemo } from "react";
import { Input } from "@/components/ui/input";
import CodeBlock from "@/app/(dashboard)/api-reference/components/CodeBlock";

const HowItWorks: React.FC = () => {
  const [responseCost, setResponseCost] = useState("");
  const [discountAmount, setDiscountAmount] = useState("");

  const calculatedDiscount = useMemo(() => {
    const cost = parseFloat(responseCost);
    const discount = parseFloat(discountAmount);

    if (isNaN(cost) || isNaN(discount) || cost === 0 || discount === 0) {
      return null;
    }

    const originalCost = cost + discount;
    const discountPercentage = (discount / originalCost) * 100;

    return {
      originalCost: originalCost.toFixed(10),
      finalCost: cost.toFixed(10),
      discountAmount: discount.toFixed(10),
      discountPercentage: discountPercentage.toFixed(2),
    };
  }, [responseCost, discountAmount]);

  return (
    <div className="space-y-4 pt-2">
      <div>
        <p className="font-medium text-foreground text-sm mb-1">
          Cost Calculation
        </p>
        <p className="text-xs text-muted-foreground">
          Discounts are applied to provider costs:{" "}
          <code className="bg-muted px-1.5 py-0.5 rounded text-xs">
            final_cost = base_cost × (1 - discount%/100)
          </code>
        </p>
      </div>
      <div>
        <p className="font-medium text-foreground text-sm mb-1">Example</p>
        <p className="text-xs text-muted-foreground">
          A 5% discount on a $10.00 request results in: $10.00 × (1 - 0.05) =
          $9.50
        </p>
      </div>
      <div>
        <p className="font-medium text-foreground text-sm mb-1">Valid Range</p>
        <p className="text-xs text-muted-foreground">
          Discount percentages must be between 0% and 100%
        </p>
      </div>

      <div className="pt-4 border-t border-border">
        <p className="font-medium text-foreground text-sm mb-2">
          Validating Discounts
        </p>
        <p className="text-xs text-muted-foreground mb-3">
          Make a test request and check the response headers to verify
          discounts are applied:
        </p>
        <CodeBlock
          language="bash"
          code={`curl -X POST -i http://your-proxy:4000/chat/completions \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer sk-1234" \\
  -d '{
    "model": "gemini/gemini-2.5-pro",
    "messages": [{"role": "user", "content": "Hello"}]
  }'`}
        />
        <p className="text-xs text-muted-foreground mt-3 mb-2">
          Look for these headers in the response:
        </p>
        <div className="space-y-1.5">
          <div className="flex items-start gap-3">
            <code className="bg-muted px-2 py-1 rounded text-xs font-mono text-foreground whitespace-nowrap">
              x-litellm-response-cost
            </code>
            <span className="text-xs text-muted-foreground">
              Final cost after discount
            </span>
          </div>
          <div className="flex items-start gap-3">
            <code className="bg-muted px-2 py-1 rounded text-xs font-mono text-foreground whitespace-nowrap">
              x-litellm-response-cost-original
            </code>
            <span className="text-xs text-muted-foreground">
              Original cost before discount
            </span>
          </div>
          <div className="flex items-start gap-3">
            <code className="bg-muted px-2 py-1 rounded text-xs font-mono text-foreground whitespace-nowrap">
              x-litellm-response-cost-discount-amount
            </code>
            <span className="text-xs text-muted-foreground">
              Amount discounted
            </span>
          </div>
        </div>
      </div>

      <div className="pt-4 border-t border-border">
        <p className="font-medium text-foreground text-sm mb-3">
          Discount Calculator
        </p>
        <p className="text-xs text-muted-foreground mb-3">
          Enter values from your response headers to verify the discount:
        </p>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
          <div>
            <label className="block text-xs font-medium text-foreground mb-1">
              Response Cost (x-litellm-response-cost)
            </label>
            <Input
              placeholder="0.0171938125"
              value={responseCost}
              onChange={(e) => setResponseCost(e.target.value)}
              className="text-sm"
            />
          </div>
          <div>
            <label className="block text-xs font-medium text-foreground mb-1">
              Discount Amount (x-litellm-response-cost-discount-amount)
            </label>
            <Input
              placeholder="0.0009049375"
              value={discountAmount}
              onChange={(e) => setDiscountAmount(e.target.value)}
              className="text-sm"
            />
          </div>
        </div>

        {calculatedDiscount && (
          <div className="bg-blue-50 border border-blue-200 dark:bg-blue-950/30 dark:border-blue-900 rounded-lg p-4">
            <p className="text-sm font-medium text-blue-900 dark:text-blue-200 mb-2">
              Calculated Results
            </p>
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-xs text-blue-800 dark:text-blue-300">
                  Original Cost:
                </span>
                <code className="text-xs font-mono text-blue-900 dark:text-blue-200">
                  ${calculatedDiscount.originalCost}
                </code>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-xs text-blue-800 dark:text-blue-300">
                  Final Cost:
                </span>
                <code className="text-xs font-mono text-blue-900 dark:text-blue-200">
                  ${calculatedDiscount.finalCost}
                </code>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-xs text-blue-800 dark:text-blue-300">
                  Discount Amount:
                </span>
                <code className="text-xs font-mono text-blue-900 dark:text-blue-200">
                  ${calculatedDiscount.discountAmount}
                </code>
              </div>
              <div className="flex items-center justify-between pt-2 border-t border-blue-300 dark:border-blue-800">
                <span className="text-xs font-semibold text-blue-900 dark:text-blue-200">
                  Discount Applied:
                </span>
                <span className="text-sm font-bold text-blue-900 dark:text-blue-200">
                  {calculatedDiscount.discountPercentage}%
                </span>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default HowItWorks;
