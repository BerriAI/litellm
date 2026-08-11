import React, { useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import { Input } from "antd";
import { SearchOutlined, ArrowRightOutlined } from "@ant-design/icons";
import { GuardrailCardInfo, ALL_CARDS } from "./guardrail_garden_data";
import GuardrailCard from "./guardrail_garden_card";
import GuardrailDetailView from "./guardrail_garden_detail";

interface GuardrailGardenProps {
  accessToken: string | null;
  onGuardrailCreated: () => void;
}

const GuardrailGarden: React.FC<GuardrailGardenProps> = ({ accessToken, onGuardrailCreated }) => {
  const { t } = useTranslation("gateway");
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedCard, setSelectedCard] = useState<GuardrailCardInfo | null>(null);
  const [showAllLitellm, setShowAllLitellm] = useState(false);
  const CARDS_PER_ROW = 5;
  const VISIBLE_ROWS = 2;

  const localizedCards = useMemo(
    () =>
      ALL_CARDS.map((card) => ({
        ...card,
        name: String(
          t(`guardrailsPage.garden.cards.${card.id}.name`, {
            defaultValue: card.name,
          }),
        ),
        description: String(
          t(`guardrailsPage.garden.cards.${card.id}.description`, {
            defaultValue: card.description,
          }),
        ),
        subcategory: card.subcategory
          ? String(
              t(`guardrailsPage.garden.labels.${card.subcategory}`, {
                defaultValue: card.subcategory,
              }),
            )
          : undefined,
        tags: card.tags.map((tag) =>
          String(
            t(`guardrailsPage.garden.labels.${tag}`, {
              defaultValue: tag,
            }),
          ),
        ),
      })),
    [t],
  );

  const filteredCards = localizedCards.filter((card) => {
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
          placeholder={t("guardrailsPage.garden.searchPlaceholder")}
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
            {t("guardrailsPage.garden.litellmTitle")}
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
              <>{t("guardrailsPage.garden.showLess")}</>
            ) : (
              <>
                <ArrowRightOutlined style={{ fontSize: 12 }} />
                {t("guardrailsPage.garden.showAll", { count: litellmCards.length })}
              </>
            )}
          </span>
        </div>
        <p style={{ fontSize: 13, color: "#6b7280", margin: "4px 0 20px 0" }}>
          {t("guardrailsPage.garden.litellmDescription")}
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
          {t("guardrailsPage.garden.partnerTitle")}
        </h2>
        <p style={{ fontSize: 13, color: "#6b7280", margin: "4px 0 20px 0" }}>
          {t("guardrailsPage.garden.partnerDescription")}
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
