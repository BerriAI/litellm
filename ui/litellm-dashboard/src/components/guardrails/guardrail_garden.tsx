import React, { useState } from "react";
import { Input } from "@/components/ui/input";
import { ArrowRight, Search } from "lucide-react";
import { GuardrailCardInfo, ALL_CARDS } from "./guardrail_garden_data";
import GuardrailCard from "./guardrail_garden_card";
import GuardrailDetailView from "./guardrail_garden_detail";

interface GuardrailGardenProps {
  accessToken: string | null;
  onGuardrailCreated: () => void;
}

const GuardrailGarden: React.FC<GuardrailGardenProps> = ({
  accessToken,
  onGuardrailCreated,
}) => {
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedCard, setSelectedCard] = useState<GuardrailCardInfo | null>(
    null,
  );
  const [showAllLitellm, setShowAllLitellm] = useState(false);
  const CARDS_PER_ROW = 5;
  const VISIBLE_ROWS = 2;

  const filteredCards = ALL_CARDS.filter((card) => {
    if (!searchQuery) return true;
    const q = searchQuery.toLowerCase();
    return (
      card.name.toLowerCase().includes(q) ||
      card.description.toLowerCase().includes(q) ||
      card.tags.some((t) => t.toLowerCase().includes(q))
    );
  });

  const litellmCards = filteredCards.filter((c) => c.category === "litellm");
  const partnerCards = filteredCards.filter((c) => c.category === "partner");

  if (selectedCard) {
    return (
      <GuardrailDetailView
        card={selectedCard}
        onBack={() => setSelectedCard(null)}
        accessToken={accessToken}
        onGuardrailCreated={onGuardrailCreated}
      />
    );
  }

  return (
    <div>
      <div className="mb-6">
        <div className="relative">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground pointer-events-none" />
          <Input
            placeholder="Search guardrails"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="pl-9 h-11 rounded-lg"
          />
        </div>
      </div>

      <div className="mb-10">
        <div className="flex items-center justify-between mb-1">
          <h2 className="text-xl font-semibold text-foreground m-0">
            LiteLLM Content Filter
          </h2>
          <button
            type="button"
            onClick={() => setShowAllLitellm(!showAllLitellm)}
            className="inline-flex items-center gap-1.5 text-sm text-primary cursor-pointer bg-transparent border-none"
          >
            {showAllLitellm ? (
              <>Show less</>
            ) : (
              <>
                <ArrowRight className="h-3 w-3" />
                {`Show all (${litellmCards.length})`}
              </>
            )}
          </button>
        </div>
        <p className="text-sm text-muted-foreground mt-1 mb-5">
          Built-in guardrails powered by LiteLLM. Zero latency, no external
          dependencies, no additional cost.
        </p>
        <div
          className="grid gap-4"
          style={{
            gridTemplateColumns:
              "repeat(auto-fill, minmax(220px, 1fr))",
          }}
        >
          {(showAllLitellm
            ? litellmCards
            : litellmCards.slice(0, CARDS_PER_ROW * VISIBLE_ROWS)
          ).map((card) => (
            <GuardrailCard
              key={card.id}
              card={card}
              onClick={() => setSelectedCard(card)}
            />
          ))}
        </div>
      </div>

      <div className="mb-10">
        <h2 className="text-xl font-semibold text-foreground mt-0 mb-1">
          Partner Guardrails
        </h2>
        <p className="text-sm text-muted-foreground mt-1 mb-5">
          Third-party guardrail integrations from leading AI security
          providers.
        </p>
        <div
          className="grid gap-4"
          style={{
            gridTemplateColumns:
              "repeat(auto-fill, minmax(220px, 1fr))",
          }}
        >
          {partnerCards.map((card) => (
            <GuardrailCard
              key={card.id}
              card={card}
              onClick={() => setSelectedCard(card)}
            />
          ))}
        </div>
      </div>
    </div>
  );
};

export default GuardrailGarden;
