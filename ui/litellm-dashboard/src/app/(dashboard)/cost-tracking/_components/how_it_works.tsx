import React, { useState, useMemo } from "react";
import CodeBlock from "@/components/CodeBlock";
import { Input } from "@/components/ui/input";

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
        <p className="mb-1 text-sm font-medium text-foreground">Cost Calculation</p>
        <p className="text-xs text-muted-foreground">
          Discounts are applied to provider costs:{" "}
          <code className="rounded-sm bg-muted px-1.5 py-0.5 text-xs">
            final_cost = base_cost × (1 - discount%/100)
          </code>
        </p>
      </div>
      <div>
        <p className="mb-1 text-sm font-medium text-foreground">Example</p>
        <p className="text-xs text-muted-foreground">
          A 5% discount on a $10.00 request results in: $10.00 × (1 - 0.05) = $9.50
        </p>
      </div>
      <div>
        <p className="mb-1 text-sm font-medium text-foreground">Valid Range</p>
        <p className="text-xs text-muted-foreground">Discount percentages must be between 0% and 100%</p>
      </div>

      <div className="border-t border-border pt-4">
        <p className="mb-2 text-sm font-medium text-foreground">Validating Discounts</p>
        <p className="mb-3 text-xs text-muted-foreground">
          Make a test request and check the response headers to verify discounts are applied:
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
        <p className="mt-3 mb-2 text-xs text-muted-foreground">Look for these headers in the response:</p>
        <div className="space-y-1.5">
          <div className="flex items-start gap-3">
            <code className="rounded-sm bg-muted px-2 py-1 font-mono text-xs whitespace-nowrap">
              x-litellm-response-cost
            </code>
            <p className="text-xs text-muted-foreground">Final cost after discount</p>
          </div>
          <div className="flex items-start gap-3">
            <code className="rounded-sm bg-muted px-2 py-1 font-mono text-xs whitespace-nowrap">
              x-litellm-response-cost-original
            </code>
            <p className="text-xs text-muted-foreground">Original cost before discount</p>
          </div>
          <div className="flex items-start gap-3">
            <code className="rounded-sm bg-muted px-2 py-1 font-mono text-xs whitespace-nowrap">
              x-litellm-response-cost-discount-amount
            </code>
            <p className="text-xs text-muted-foreground">Amount discounted</p>
          </div>
        </div>
      </div>

      <div className="border-t border-border pt-4">
        <p className="mb-3 text-sm font-medium text-foreground">Discount Calculator</p>
        <p className="mb-3 text-xs text-muted-foreground">
          Enter values from your response headers to verify the discount:
        </p>
        <div className="mb-4 grid grid-cols-1 gap-4 md:grid-cols-2">
          <div>
            <label className="mb-1 block text-xs font-medium text-foreground" htmlFor="how-it-works-response-cost">
              Response Cost (x-litellm-response-cost)
            </label>
            <Input
              id="how-it-works-response-cost"
              placeholder="0.0171938125"
              value={responseCost}
              onChange={(event) => setResponseCost(event.target.value)}
              className="text-sm"
            />
          </div>
          <div>
            <label className="mb-1 block text-xs font-medium text-foreground" htmlFor="how-it-works-discount-amount">
              Discount Amount (x-litellm-response-cost-discount-amount)
            </label>
            <Input
              id="how-it-works-discount-amount"
              placeholder="0.0009049375"
              value={discountAmount}
              onChange={(event) => setDiscountAmount(event.target.value)}
              className="text-sm"
            />
          </div>
        </div>

        {calculatedDiscount && (
          <div className="rounded-lg border border-border bg-muted p-4">
            <p className="mb-2 text-sm font-medium text-foreground">Calculated Results</p>
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <p className="text-xs text-muted-foreground">Original Cost:</p>
                <code className="font-mono text-xs text-foreground">${calculatedDiscount.originalCost}</code>
              </div>
              <div className="flex items-center justify-between">
                <p className="text-xs text-muted-foreground">Final Cost:</p>
                <code className="font-mono text-xs text-foreground">${calculatedDiscount.finalCost}</code>
              </div>
              <div className="flex items-center justify-between">
                <p className="text-xs text-muted-foreground">Discount Amount:</p>
                <code className="font-mono text-xs text-foreground">${calculatedDiscount.discountAmount}</code>
              </div>
              <div className="flex items-center justify-between border-t border-border pt-2">
                <p className="text-xs font-semibold text-foreground">Discount Applied:</p>
                <p className="text-sm font-bold text-foreground">{calculatedDiscount.discountPercentage}%</p>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default HowItWorks;
