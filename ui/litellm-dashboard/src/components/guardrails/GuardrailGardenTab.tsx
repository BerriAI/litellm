"use client";

import React, { useState } from "react";
import {
  SearchIcon,
  ArrowRightIcon,
  CheckCircleIcon,
  ShieldIcon,
} from "lucide-react";

const MOCK_GARDEN_CARDS = [
  {
    title: "Denied Financial Advice",
    description:
      "Detects requests for personalized financial advice, investment recommendations, or financial...",
    f1Score: "100%",
    testCases: 207,
  },
  {
    title: "Insults & Personal Attacks",
    description:
      "Detects insults, name-calling, and personal attacks directed at the chatbot, staff, or other people.",
    f1Score: "100%",
    testCases: 299,
  },
  {
    title: "Denied Legal Advice",
    description:
      "Detects requests for unauthorized legal advice, case analysis, or legal recommendations.",
  },
  {
    title: "Denied Medical Advice",
    description:
      "Detects requests for medical diagnosis, treatment recommendations, or health advice.",
  },
  {
    title: "Harmful Violence",
    description:
      "Detects content related to violence, criminal planning, attacks, and violent threats.",
  },
  {
    title: "Harmful Self-Harm",
    description:
      "Detects content related to self-harm, suicide, and dangerous self-destructive behavior.",
  },
  {
    title: "Harmful Child Safety",
    description:
      "Detects content that could endanger child safety or exploit minors.",
  },
  {
    title: "Harmful Illegal Weapons",
    description:
      "Detects content related to illegal weapons manufacturing, distribution, or acquisition.",
  },
  {
    title: "Bias: Gender",
    description:
      "Detects gender-based discrimination, stereotypes, and biased language.",
  },
  {
    title: "Bias: Racial",
    description:
      "Detects racial discrimination, stereotypes, and racially biased content.",
  },
];

type GuardrailCardProps = {
  title: string;
  description: string;
  f1Score?: string;
  testCases?: number;
};

function GuardrailCard({
  title,
  description,
  f1Score,
  testCases,
}: GuardrailCardProps) {
  return (
    <div className="border border-gray-200 rounded-lg p-4 bg-white hover:border-gray-300 transition-colors cursor-pointer">
      <div className="flex items-start gap-3 mb-2">
        <div className="flex-shrink-0 w-9 h-9 rounded-full bg-blue-100 flex items-center justify-center">
          <ShieldIcon className="h-4 w-4 text-blue-500" />
        </div>
        <h3 className="text-sm font-semibold text-gray-900 leading-tight">
          {title}
        </h3>
      </div>
      <p className="text-xs text-gray-500 leading-relaxed mb-2">
        {description}
      </p>
      {f1Score && testCases !== undefined && (
        <div className="flex items-center gap-1 text-xs text-green-600">
          <CheckCircleIcon className="h-3.5 w-3.5" />
          <span>
            F1: {f1Score} · {testCases} test cases
          </span>
        </div>
      )}
    </div>
  );
}

export function GuardrailGardenTab() {
  const [search, setSearch] = useState("");

  const filteredCards = MOCK_GARDEN_CARDS.filter(
    (card) =>
      !search ||
      card.title.toLowerCase().includes(search.toLowerCase()) ||
      card.description.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <div className="p-6">
      {/* Search */}
      <div className="relative mb-8">
        <SearchIcon className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-gray-400" />
        <input
          type="text"
          placeholder="Search guardrails"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="w-full pl-9 pr-4 py-2 border border-gray-200 rounded-md text-sm text-gray-700 placeholder-gray-400 focus:outline-none focus:ring-1 focus:ring-blue-500 focus:border-blue-500"
        />
      </div>

      {/* LiteLLM Content Filter Section */}
      <div className="mb-6">
        <div className="flex items-center justify-between mb-1">
          <h2 className="text-lg font-semibold text-gray-900">
            LiteLLM Content Filter
          </h2>
          <button
            type="button"
            className="text-sm text-blue-500 hover:text-blue-600 flex items-center gap-1"
          >
            <ArrowRightIcon className="h-3.5 w-3.5" />
            Show all ({MOCK_GARDEN_CARDS.length})
          </button>
        </div>
        <p className="text-sm text-gray-500 mb-5">
          Built-in guardrails powered by LiteLLM. Zero latency, no external
          dependencies, no additional cost.
        </p>

        {/* Row 1 */}
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3 mb-3">
          {filteredCards.slice(0, 6).map((card) => (
            <GuardrailCard
              key={card.title}
              title={card.title}
              description={card.description}
              f1Score={card.f1Score}
              testCases={card.testCases}
            />
          ))}
        </div>

        {/* Row 2 */}
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
          {filteredCards.slice(6, 10).map((card) => (
            <GuardrailCard
              key={card.title}
              title={card.title}
              description={card.description}
              f1Score={card.f1Score}
              testCases={card.testCases}
            />
          ))}
        </div>
      </div>
    </div>
  );
}
