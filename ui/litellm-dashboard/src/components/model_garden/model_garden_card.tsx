import React, { useState } from "react";
import { ProviderGroup, ModelInfo, formatCost, formatContextWindow, getCapabilities } from "./model_garden_utils";

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

export const ProviderCard: React.FC<{
  group: ProviderGroup;
  onClick: () => void;
}> = ({ group, onClick }) => {
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
      <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 10 }}>
        <LogoWithFallback src={group.logo} name={group.displayName} />
        <span style={{ fontSize: 14, fontWeight: 600, color: "#111827", lineHeight: 1.3 }}>
          {group.displayName}
        </span>
      </div>

      <p
        className="line-clamp-2"
        style={{ fontSize: 12, color: "#6b7280", lineHeight: 1.6, margin: 0, flex: 1 }}
      >
        {group.modelCount} models &middot; {group.modes.slice(0, 3).join(", ")}
        {group.modes.length > 3 && ` +${group.modes.length - 3} more`}
      </p>

      {group.capabilities.length > 0 && (
        <div style={{ marginTop: 10, display: "flex", flexWrap: "wrap", gap: 4 }}>
          {group.capabilities.slice(0, 3).map((cap) => (
            <span
              key={cap}
              style={{
                fontSize: 11,
                padding: "2px 8px",
                borderRadius: 12,
                backgroundColor: "#f0f9ff",
                color: "#1d4ed8",
                fontWeight: 500,
              }}
            >
              {cap}
            </span>
          ))}
          {group.capabilities.length > 3 && (
            <span
              style={{
                fontSize: 11,
                padding: "2px 8px",
                borderRadius: 12,
                backgroundColor: "#f3f4f6",
                color: "#6b7280",
                fontWeight: 500,
              }}
            >
              +{group.capabilities.length - 3}
            </span>
          )}
        </div>
      )}
    </div>
  );
};

export const NewModelCard: React.FC<{
  model: ModelInfo;
  providerLogo: string;
  providerName: string;
  onClick: () => void;
}> = ({ model, providerLogo, providerName, onClick }) => {
  const [hovered, setHovered] = useState(false);
  const caps = getCapabilities(model);

  const baseName = model.key.includes("/")
    ? model.key.split("/").slice(-1)[0]
    : model.key;

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
      <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 10 }}>
        <LogoWithFallback src={providerLogo} name={providerName} />
        <div style={{ minWidth: 0 }}>
          <span style={{ fontSize: 14, fontWeight: 600, color: "#111827", lineHeight: 1.3, display: "block" }}>
            {baseName}
          </span>
          <span style={{ fontSize: 11, color: "#9ca3af" }}>{providerName}</span>
        </div>
      </div>

      <div style={{ fontSize: 12, color: "#6b7280", lineHeight: 1.8, flex: 1 }}>
        {model.input_cost_per_token !== undefined && (
          <div>
            Input: {formatCost(model.input_cost_per_token)} &middot; Output: {formatCost(model.output_cost_per_token)}
          </div>
        )}
        {model.max_input_tokens && (
          <div>Context: {formatContextWindow(model.max_input_tokens)}</div>
        )}
      </div>

      {caps.length > 0 && (
        <div style={{ marginTop: 10, display: "flex", flexWrap: "wrap", gap: 4 }}>
          {caps.slice(0, 3).map((cap) => (
            <span
              key={cap}
              style={{
                fontSize: 11,
                padding: "2px 8px",
                borderRadius: 12,
                backgroundColor: "#f0fdf4",
                color: "#15803d",
                fontWeight: 500,
              }}
            >
              {cap}
            </span>
          ))}
          {caps.length > 3 && (
            <span
              style={{
                fontSize: 11,
                padding: "2px 8px",
                borderRadius: 12,
                backgroundColor: "#f3f4f6",
                color: "#6b7280",
                fontWeight: 500,
              }}
            >
              +{caps.length - 3}
            </span>
          )}
        </div>
      )}
    </div>
  );
};
