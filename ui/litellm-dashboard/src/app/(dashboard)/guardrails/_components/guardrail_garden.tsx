import React, { useEffect } from "react";
import { ArrowRight, Search } from "lucide-react";
import { parseAsBoolean, parseAsString, useQueryState, useQueryStates } from "nuqs";
import { InputGroup, InputGroupAddon, InputGroupInput } from "@/components/ui/input-group";
import { ALL_CARDS } from "./guardrail_garden_data";
import GuardrailCard from "./guardrail_garden_card";
import GuardrailDetailView from "./guardrail_garden_detail";
import { GARDEN_TAB_KEY } from "./useGardenDetailTab";

const searchParser = parseAsString.withDefault("");
const showAllParser = parseAsBoolean.withDefault(false);
const selectedCardParsers = {
  garden_card: parseAsString.withOptions({ history: "push" }),
  [GARDEN_TAB_KEY]: parseAsString,
};

interface GuardrailGardenProps {
  accessToken: string | null;
  onGuardrailCreated: () => void;
}

const GuardrailGarden: React.FC<GuardrailGardenProps> = ({ accessToken, onGuardrailCreated }) => {
  const [searchQuery, setSearchQuery] = useQueryState("garden_q", searchParser);
  const [showAllLitellm, setShowAllLitellm] = useQueryState("garden_all", showAllParser);
  const [{ garden_card: selectedCardId }, setSelectedCardState] = useQueryStates(selectedCardParsers);
  const selectedCard = ALL_CARDS.find((card) => card.id === selectedCardId) ?? null;
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

  const unknownCardSelected = selectedCardId !== null && selectedCard === null;
  useEffect(() => {
    if (unknownCardSelected) void setSelectedCardState(null, { history: "replace" });
  }, [unknownCardSelected, setSelectedCardState]);

  const openCard = (cardId: string) => void setSelectedCardState({ garden_card: cardId, [GARDEN_TAB_KEY]: null });
  const closeCard = () => void setSelectedCardState(null, { history: "replace" });

  if (selectedCard) {
    return (
      <GuardrailDetailView
        card={selectedCard}
        onBack={closeCard}
        accessToken={accessToken}
        onGuardrailCreated={onGuardrailCreated}
      />
    );
  }

  return (
    <div>
      <div className="mb-6">
        <InputGroup>
          <InputGroupAddon>
            <Search className="size-4 text-muted-foreground" />
          </InputGroupAddon>
          <InputGroupInput
            placeholder="Search guardrails"
            value={searchQuery}
            onChange={(e) => void setSearchQuery(e.target.value)}
          />
        </InputGroup>
      </div>

      <div className="mb-10">
        <div className="mb-1 flex items-center justify-between">
          <h2 className="m-0 text-xl font-semibold text-foreground">LiteLLM Content Filter</h2>
          <span
            className="inline-flex cursor-pointer items-center gap-1.5 text-sm text-primary"
            onClick={() => void setShowAllLitellm(!showAllLitellm)}
          >
            {showAllLitellm ? (
              <>Show less</>
            ) : (
              <>
                <ArrowRight className="size-3" />
                {`Show all (${litellmCards.length})`}
              </>
            )}
          </span>
        </div>
        <p className="mt-1 mb-5 text-[13px] text-muted-foreground">
          Built-in guardrails powered by LiteLLM. Zero latency, no external dependencies, no additional cost.
        </p>
        <div className="grid grid-cols-[repeat(auto-fill,minmax(220px,1fr))] gap-4">
          {(showAllLitellm ? litellmCards : litellmCards.slice(0, CARDS_PER_ROW * VISIBLE_ROWS)).map((card) => (
            <GuardrailCard key={card.id} card={card} onClick={() => openCard(card.id)} />
          ))}
        </div>
      </div>

      <div className="mb-10">
        <h2 className="mt-0 mb-1 text-xl font-semibold text-foreground">Partner Guardrails</h2>
        <p className="mt-1 mb-5 text-[13px] text-muted-foreground">
          Third-party guardrail integrations from leading AI security providers.
        </p>
        <div className="grid grid-cols-[repeat(auto-fill,minmax(220px,1fr))] gap-4">
          {partnerCards.map((card) => (
            <GuardrailCard key={card.id} card={card} onClick={() => openCard(card.id)} />
          ))}
        </div>
      </div>
    </div>
  );
};

export default GuardrailGarden;
