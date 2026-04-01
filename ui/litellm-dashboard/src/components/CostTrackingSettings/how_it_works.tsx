import React, { useState, useMemo } from "react";
import { Text, TextInput } from "@tremor/react";
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
            <Text className="font-medium text-gray-900 text-sm mb-1">Cost Calculation</Text>
            <Text className="text-xs text-gray-600">
              Discounts are applied to provider costs: <code className="bg-gray-100 px-1.5 py-0.5 rounded text-xs">final_cost = base_cost × (1 - discount%/100)</code>
            </Text>
          </div>
          <div>
            <Text className="font-medium text-gray-900 text-sm mb-1">Example</Text>
            <Text className="text-xs text-gray-600">
              A 5% discount on a $10.00 request results in: $10.00 × (1 - 0.05) = $9.50
            </Text>
          </div>
          <div>
            <Text className="font-medium text-gray-900 text-sm mb-1">Valid Range</Text>
            <Text className="text-xs text-gray-600">
              Discount percentages must be between 0% and 100%
            </Text>
          </div>

          <div className="pt-4 border-t border-gray-200">
            <Text className="font-medium text-gray-900 text-sm mb-2">Validating Discounts</Text>
            <Text className="text-xs text-gray-600 mb-3">
              Make a test request and check the response headers to verify discounts are applied:
            </Text>
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
            <Text className="text-xs text-gray-600 mt-3 mb-2">
              Look for these headers in the response:
            </Text>
            <div className="space-y-1.5">
              <div className="flex items-start gap-3">
                <code className="bg-gray-100 px-2 py-1 rounded text-xs font-mono text-gray-800 whitespace-nowrap">
                  x-litellm-response-cost
                </code>
                <Text className="text-xs text-gray-600">Final cost after discount</Text>
              </div>
              <div className="flex items-start gap-3">
                <code className="bg-gray-100 px-2 py-1 rounded text-xs font-mono text-gray-800 whitespace-nowrap">
                  x-litellm-response-cost-original
                </code>
                <Text className="text-xs text-gray-600">Original cost before discount</Text>
              </div>
              <div className="flex items-start gap-3">
                <code className="bg-gray-100 px-2 py-1 rounded text-xs font-mono text-gray-800 whitespace-nowrap">
                  x-litellm-response-cost-discount-amount
                </code>
                <Text className="text-xs text-gray-600">Amount discounted</Text>
              </div>
            </div>
          </div>

          <div className="pt-4 border-t border-gray-200">
            <Text className="font-medium text-gray-900 text-sm mb-3">Discount Calculator</Text>
            <Text className="text-xs text-gray-600 mb-3">
              Enter values from your response headers to verify the discount:
            </Text>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
              <div>
                <label className="block text-xs font-medium text-gray-700 mb-1">
                  Response Cost (x-litellm-response-cost)
                </label>
                <TextInput
                  placeholder="0.0171938125"
                  value={responseCost}
                  onValueChange={setResponseCost}
                  className="text-sm"
                />
              </div>
              <div>
                <label className="block text-xs font-medium text-gray-700 mb-1">
                  Discount Amount (x-litellm-response-cost-discount-amount)
                </label>
                <TextInput
                  placeholder="0.0009049375"
                  value={discountAmount}
                  onValueChange={setDiscountAmount}
                  className="text-sm"
                />
              </div>
            </div>

            {calculatedDiscount && (
              <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                <Text className="text-sm font-medium text-blue-900 mb-2">Calculated Results</Text>
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <Text className="text-xs text-blue-800">Original Cost:</Text>
                    <code className="text-xs font-mono text-blue-900">${calculatedDiscount.originalCost}</code>
                  </div>
                  <div className="flex items-center justify-between">
                    <Text className="text-xs text-blue-800">Final Cost:</Text>
                    <code className="text-xs font-mono text-blue-900">${calculatedDiscount.finalCost}</code>
                  </div>
                  <div className="flex items-center justify-between">
                    <Text className="text-xs text-blue-800">Discount Amount:</Text>
                    <code className="text-xs font-mono text-blue-900">${calculatedDiscount.discountAmount}</code>
                  </div>
                  <div className="flex items-center justify-between pt-2 border-t border-blue-300">
                    <Text className="text-xs font-semibold text-blue-900">Discount Applied:</Text>
                    <Text className="text-sm font-bold text-blue-900">{calculatedDiscount.discountPercentage}%</Text>
                  </div>
                </div>
              </div>
            )}
          </div>
    </div>
  );
};

export default HowItWorks;

