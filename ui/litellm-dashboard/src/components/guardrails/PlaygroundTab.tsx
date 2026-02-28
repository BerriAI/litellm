"use client";

import React, { useState } from "react";

export function PlaygroundTab() {
  const [prompt, setPrompt] = useState("");
  const [result, setResult] = useState<string | null>(null);

  function handleTest() {
    setResult(
      prompt.toLowerCase().includes("financial")
        ? "🚫 Blocked by: Denied Financial Advice guardrail (confidence: 97%)"
        : "✅ Passed all guardrails. Safe to proceed."
    );
  }

  return (
    <div className="p-6 max-w-2xl">
      <div className="mb-4">
        <h2 className="text-lg font-semibold text-gray-900 mb-1">
          Test Playground
        </h2>
        <p className="text-sm text-gray-500">
          Test your guardrails against sample prompts to verify they work as
          expected.
        </p>
      </div>
      <div className="space-y-4">
        <div>
          <label className="block text-xs font-semibold text-gray-700 mb-1.5">
            Test Prompt
          </label>
          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="Enter a prompt to test against your guardrails..."
            rows={4}
            className="w-full border border-gray-200 rounded-lg px-3 py-2.5 text-sm text-gray-700 placeholder-gray-400 focus:outline-none focus:ring-1 focus:ring-blue-500 focus:border-blue-500 resize-none"
          />
        </div>
        <button
          type="button"
          onClick={handleTest}
          disabled={!prompt.trim()}
          className="bg-blue-500 hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed text-white text-sm font-medium px-4 py-2 rounded-md transition-colors"
        >
          Run Test
        </button>
        {result && (
          <div
            className={`border rounded-lg px-4 py-3 text-sm font-medium ${
              result.startsWith("🚫")
                ? "border-red-200 bg-red-50 text-red-700"
                : "border-green-200 bg-green-50 text-green-700"
            }`}
          >
            {result}
          </div>
        )}
      </div>
    </div>
  );
}
