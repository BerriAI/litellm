import React, { useState } from "react";
import { CheckCircleFilled } from "@ant-design/icons";
import { GuardrailCardInfo } from "./guardrail_garden_data";

const LogoWithFallback: React.FC<{ src: string; name: string }> = ({ src, name }) => {
  const [hasError, setHasError] = useState(false);

  if (hasError || !src) {
    return (
      <div
        style={{
          width: 28,
          height: 28,
          borderRadius: 6,
          backgroundColor: "#e5e7eb",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          fontSize: 13,
          fontWeight: 600,
          color: "#6b7280",
          flexShrink: 0,
        }}
      >
        {name?.charAt(0) || "?"}
      </div>
    );
  }

  return (
    <img
      src={src}
      alt=""
      style={{ width: 28, height: 28, borderRadius: 6, objectFit: "contain", flexShrink: 0 }}
      onError={() => setHasError(true)}
    />
  );
};

const GuardrailCard: React.FC<{ card: GuardrailCardInfo; onClick: () => void }> = ({ card, onClick }) => {
  const [hovered, setHovered] = useState(false);

  return (
    <div
      onClick={onClick}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        borderRadius: 12,
        border: hovered ? "1px solid #93c5fd" : "1px solid #e5e7eb",
        backgroundColor: "#ffffff",
        padding: "20px 20px 16px 20px",
        cursor: "pointer",
        transition: "border-color 0.15s, box-shadow 0.15s",
        display: "flex",
        flexDirection: "column",
        minHeight: 170,
        boxShadow: hovered ? "0 1px 6px rgba(59,130,246,0.08)" : "none",
      }}
    >
      {/* Icon + Name row */}
      <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 10 }}>
        <LogoWithFallback src={card.logo} name={card.name} />
        <span style={{ fontSize: 14, fontWeight: 600, color: "#111827", lineHeight: 1.3 }}>{card.name}</span>
      </div>

      {/* Description */}
      <p
        className="line-clamp-3"
        style={{ fontSize: 12, color: "#6b7280", lineHeight: 1.6, margin: 0, flex: 1 }}
      >
        {card.description}
      </p>

      {/* Eval badge */}
      {card.eval && (
        <div style={{ marginTop: 10, display: "flex", alignItems: "center", gap: 4 }}>
          <CheckCircleFilled style={{ color: "#16a34a", fontSize: 12 }} />
          <span style={{ fontSize: 11, color: "#16a34a", fontWeight: 500 }}>
            F1: {card.eval.f1}% &middot; {card.eval.testCases} test cases
          </span>
        </div>
      )}
    </div>
  );
};

export default GuardrailCard;
