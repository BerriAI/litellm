"use client";

import React from "react";

const MOCK_GUARDRAILS = [
  {
    name: "Denied Financial Advice",
    type: "LiteLLM Built-in",
    status: "Active" as const,
    appliedTo: "All routes",
  },
  {
    name: "Insults & Personal Attacks",
    type: "LiteLLM Built-in",
    status: "Active" as const,
    appliedTo: "Customer-facing",
  },
  {
    name: "Prompt Injection Detector",
    type: "Team Custom",
    status: "Active" as const,
    appliedTo: "ML Platform team",
  },
];

export function GuardrailsListTab() {
  return (
    <div className="p-6">
      <div className="mb-4">
        <h2 className="text-lg font-semibold text-gray-900 mb-1">Guardrails</h2>
        <p className="text-sm text-gray-500">
          Configure and manage active guardrails for your AI gateway.
        </p>
      </div>
      <div className="border border-gray-200 rounded-lg overflow-hidden">
        <table className="w-full text-sm">
          <thead>
            <tr className="bg-gray-50 border-b border-gray-200">
              <th className="text-left px-4 py-3 text-xs font-semibold text-gray-500 uppercase tracking-wide">
                Name
              </th>
              <th className="text-left px-4 py-3 text-xs font-semibold text-gray-500 uppercase tracking-wide">
                Type
              </th>
              <th className="text-left px-4 py-3 text-xs font-semibold text-gray-500 uppercase tracking-wide">
                Status
              </th>
              <th className="text-left px-4 py-3 text-xs font-semibold text-gray-500 uppercase tracking-wide">
                Applied To
              </th>
            </tr>
          </thead>
          <tbody>
            {MOCK_GUARDRAILS.map((row, i) => (
              <tr
                key={row.name}
                className={
                  i < MOCK_GUARDRAILS.length - 1
                    ? "border-b border-gray-100"
                    : ""
                }
              >
                <td className="px-4 py-3 font-medium text-gray-900">
                  {row.name}
                </td>
                <td className="px-4 py-3 text-gray-500">{row.type}</td>
                <td className="px-4 py-3">
                  <span className="inline-flex items-center gap-1.5 text-xs font-medium px-2 py-0.5 rounded-full bg-green-50 text-green-700">
                    <span className="w-1.5 h-1.5 rounded-full bg-green-500" />
                    {row.status}
                  </span>
                </td>
                <td className="px-4 py-3 text-gray-500">{row.appliedTo}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
