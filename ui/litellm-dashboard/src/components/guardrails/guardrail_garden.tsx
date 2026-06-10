import React, { useState } from "react";
import { Input } from "antd";
import { SearchOutlined, ArrowRightOutlined } from "@ant-design/icons";
import { useTranslation } from "react-i18next";
import { GuardrailCardInfo, ALL_CARDS } from "./guardrail_garden_data";
import GuardrailCard from "./guardrail_garden_card";
import GuardrailDetailView from "./guardrail_garden_detail";

interface GuardrailGardenProps {
  accessToken: string | null;
  onGuardrailCreated: () => void;
}

const GuardrailGarden: React.FC<GuardrailGardenProps> = ({ accessToken, onGuardrailCreated }) => {
  const { t } = useTranslation();
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedCard, setSelectedCard] = useState<GuardrailCardInfo | null>(null);
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
      {/* Search Bar */}
      <div style={{ marginBottom: 24 }}>
        <Input
          size="large"
          placeholder={t("guardrails.guardrailGarden.searchPlaceholder")}
          prefix={<SearchOutlined style={{ color: "#9ca3af" }} />}
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          style={{ borderRadius: 8 }}
        />
      </div>

      {/* LiteLLM Content Filter Section */}
      <div style={{ marginBottom: 40 }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 4 }}>
          <h2 style={{ fontSize: 20, fontWeight: 600, color: "#111827", margin: 0 }}>
            {t("guardrails.guardrailGarden.litellmSectionTitle")}
          </h2>
          <span
            style={{
              display: "inline-flex",
              alignItems: "center",
              gap: 6,
              fontSize: 14,
              color: "#1a73e8",
              cursor: "pointer",
            }}
            onClick={() => setShowAllLitellm(!showAllLitellm)}
          >
            {showAllLitellm ? (
              <>{t("guardrails.guardrailGarden.showLess")}</>
            ) : (
              <>
                <ArrowRightOutlined style={{ fontSize: 12 }} />
                {t("guardrails.guardrailGarden.showAll", { count: litellmCards.length })}
              </>
            )}
          </span>
        </div>
        <p style={{ fontSize: 13, color: "#6b7280", margin: "4px 0 20px 0" }}>
          {t("guardrails.guardrailGarden.litellmSectionDesc")}
        </p>
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fill, minmax(220px, 1fr))",
            gap: 16,
          }}
        >
          {(showAllLitellm ? litellmCards : litellmCards.slice(0, CARDS_PER_ROW * VISIBLE_ROWS)).map((card) => (
            <GuardrailCard key={card.id} card={card} onClick={() => setSelectedCard(card)} />
          ))}
        </div>
      </div>

      {/* Partner Guardrails Section */}
      <div style={{ marginBottom: 40 }}>
        <h2 style={{ fontSize: 20, fontWeight: 600, color: "#111827", margin: "0 0 4px 0" }}>
          {t("guardrails.guardrailGarden.partnerSectionTitle")}
        </h2>
        <p style={{ fontSize: 13, color: "#6b7280", margin: "4px 0 20px 0" }}>
          {t("guardrails.guardrailGarden.partnerSectionDesc")}
        </p>
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fill, minmax(220px, 1fr))",
            gap: 16,
          }}
        >
          {partnerCards.map((card) => (
            <GuardrailCard key={card.id} card={card} onClick={() => setSelectedCard(card)} />
          ))}
        </div>
      </div>
    </div>
  );
};

export default GuardrailGarden;
