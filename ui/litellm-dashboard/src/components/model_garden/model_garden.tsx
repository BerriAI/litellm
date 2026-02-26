import React, { useState, useMemo, useRef } from "react";
import { Input, Spin } from "antd";
import { SearchOutlined, LeftOutlined, RightOutlined, ArrowRightOutlined } from "@ant-design/icons";
import ModelGardenDetailView from "./model_garden_detail";
import {
  ProviderGroup,
  ModelInfo,
  ModelCategory,
  WhatsNewItem,
  parseModelCostMap,
  buildModelCategories,
  detectWhatsNew,
  getProviderDisplayName,
  getProviderLogo,
  getModeLabel,
  formatCost,
  formatContextWindow,
} from "./model_garden_utils";

interface ModelGardenProps {
  modelCostMap: Record<string, any> | null | undefined;
  isLoading: boolean;
}

const TINT_COLORS = ["#e8f0fe", "#e6f4ea", "#fce8e6", "#fef7e0", "#f3e8fd", "#e8eaed"];

const LogoWithFallback: React.FC<{ src: string; name: string; size?: number }> = ({ src, name, size = 32 }) => {
  const [hasError, setHasError] = useState(false);
  if (hasError || !src) {
    return (
      <div style={{
        width: size, height: size, borderRadius: 8, backgroundColor: "#e5e7eb",
        display: "flex", alignItems: "center", justifyContent: "center",
        fontSize: size * 0.45, fontWeight: 600, color: "#6b7280", flexShrink: 0,
      }}>
        {name?.charAt(0) || "?"}
      </div>
    );
  }
  return (
    <img src={src} alt="" style={{ width: size, height: size, borderRadius: 8, objectFit: "contain", flexShrink: 0 }}
      onError={() => setHasError(true)} />
  );
};

/* ── What's New announcement cards (Vertex style) ── */
const WhatsNewSection: React.FC<{ items: WhatsNewItem[] }> = ({ items }) => {
  if (items.length === 0) return null;
  return (
    <div style={{ marginBottom: 40 }}>
      <h2 style={{ fontSize: 22, fontWeight: 400, color: "#202124", margin: "0 0 20px 0" }}>
        {"What's new in Model Garden"}
      </h2>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 12 }}>
        {items.slice(0, 6).map((item, i) => (
          <div key={i} style={{
            backgroundColor: TINT_COLORS[i % TINT_COLORS.length],
            borderRadius: 12, padding: "20px 24px", cursor: "default",
            border: "1px solid transparent", transition: "border-color 0.15s",
          }}
            onMouseEnter={(e) => (e.currentTarget.style.borderColor = "#dadce0")}
            onMouseLeave={(e) => (e.currentTarget.style.borderColor = "transparent")}
          >
            <div style={{ display: "flex", alignItems: "flex-start", gap: 10, marginBottom: 8 }}>
              <div style={{
                width: 24, height: 24, borderRadius: "50%", backgroundColor: "rgba(66,133,244,0.15)",
                display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0, marginTop: 2,
              }}>
                <svg width="14" height="14" viewBox="0 0 24 24" fill="#4285f4">
                  <path d="M9 21c0 .55.45 1 1 1h4c.55 0 1-.45 1-1v-1H9v1zm3-19C8.14 2 5 5.14 5 9c0 2.38 1.19 4.47 3 5.74V17c0 .55.45 1 1 1h6c.55 0 1-.45 1-1v-2.26c1.81-1.27 3-3.36 3-5.74 0-3.86-3.14-7-7-7z" />
                </svg>
              </div>
              <div>
                <div style={{ fontSize: 14, fontWeight: 500, color: "#202124", lineHeight: 1.4 }}>{item.title}</div>
                <div style={{ fontSize: 12, color: "#5f6368", lineHeight: 1.5, marginTop: 4 }}>{item.description}</div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

/* ── Provider carousel (horizontal scroll) ── */
const ProviderCarousel: React.FC<{
  providers: ProviderGroup[];
  onSelect: (g: ProviderGroup) => void;
}> = ({ providers, onSelect }) => {
  const scrollRef = useRef<HTMLDivElement>(null);

  const scroll = (dir: "left" | "right") => {
    if (!scrollRef.current) return;
    const amount = 300;
    scrollRef.current.scrollBy({ left: dir === "left" ? -amount : amount, behavior: "smooth" });
  };

  return (
    <div style={{ marginBottom: 40 }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 16 }}>
        <h2 style={{ fontSize: 18, fontWeight: 400, color: "#202124", margin: 0 }}>All providers</h2>
        <span style={{ fontSize: 14, color: "#1a73e8", cursor: "pointer", display: "inline-flex", alignItems: "center", gap: 4 }}>
          <ArrowRightOutlined style={{ fontSize: 12 }} /> Show all ({providers.length})
        </span>
      </div>
      <div style={{ position: "relative" }}>
        <button onClick={() => scroll("left")} style={{
          position: "absolute", left: -16, top: "50%", transform: "translateY(-50%)", zIndex: 2,
          width: 36, height: 36, borderRadius: "50%", border: "1px solid #dadce0",
          backgroundColor: "#fff", cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center",
          boxShadow: "0 1px 3px rgba(0,0,0,0.12)",
        }}>
          <LeftOutlined style={{ fontSize: 12, color: "#5f6368" }} />
        </button>
        <div ref={scrollRef} style={{
          display: "flex", gap: 12, overflowX: "auto", scrollbarWidth: "none",
          padding: "4px 0", msOverflowStyle: "none",
        }}>
          {providers.map((g) => (
            <div key={g.provider} onClick={() => onSelect(g)} style={{
              minWidth: 160, padding: "20px 24px", borderRadius: 12,
              border: "1px solid #e5e7eb", backgroundColor: "#fff", cursor: "pointer",
              display: "flex", flexDirection: "column", alignItems: "center", gap: 10,
              transition: "border-color 0.15s, box-shadow 0.15s", flexShrink: 0,
            }}
              onMouseEnter={(e) => { e.currentTarget.style.borderColor = "#93c5fd"; e.currentTarget.style.boxShadow = "0 1px 6px rgba(59,130,246,0.08)"; }}
              onMouseLeave={(e) => { e.currentTarget.style.borderColor = "#e5e7eb"; e.currentTarget.style.boxShadow = "none"; }}
            >
              <LogoWithFallback src={g.logo} name={g.displayName} size={40} />
              <span style={{ fontSize: 13, fontWeight: 500, color: "#202124", textAlign: "center", lineHeight: 1.3 }}>
                {g.displayName}
              </span>
            </div>
          ))}
        </div>
        <button onClick={() => scroll("right")} style={{
          position: "absolute", right: -16, top: "50%", transform: "translateY(-50%)", zIndex: 2,
          width: 36, height: 36, borderRadius: "50%", border: "1px solid #dadce0",
          backgroundColor: "#fff", cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center",
          boxShadow: "0 1px 3px rgba(0,0,0,0.12)",
        }}>
          <RightOutlined style={{ fontSize: 12, color: "#5f6368" }} />
        </button>
      </div>
    </div>
  );
};

/* ── Model card (Vertex style: icon + name + mode) ── */
const ModelCard: React.FC<{ model: ModelInfo; onClick?: () => void }> = ({ model, onClick }) => {
  const [hovered, setHovered] = useState(false);
  const baseName = model.key.includes("/") ? model.key.split("/").slice(-1)[0] : model.key;
  const logo = getProviderLogo(model.litellm_provider);

  return (
    <div onClick={onClick}
      onMouseEnter={() => setHovered(true)} onMouseLeave={() => setHovered(false)}
      style={{
        borderRadius: 12, border: hovered ? "1px solid #93c5fd" : "1px solid #e5e7eb",
        backgroundColor: "#fff", padding: "16px 20px", cursor: onClick ? "pointer" : "default",
        display: "flex", alignItems: "center", gap: 12,
        transition: "border-color 0.15s, box-shadow 0.15s",
        boxShadow: hovered ? "0 1px 6px rgba(59,130,246,0.08)" : "none",
      }}>
      <LogoWithFallback src={logo} name={getProviderDisplayName(model.litellm_provider)} size={28} />
      <div style={{ minWidth: 0 }}>
        <div style={{ fontSize: 13, fontWeight: 500, color: "#202124", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
          {baseName}
        </div>
        <div style={{ fontSize: 11, color: "#5f6368" }}>{getModeLabel(model.mode)}</div>
      </div>
    </div>
  );
};

/* ── Category section ── */
const CategorySection: React.FC<{
  category: ModelCategory;
  onModelClick: (m: ModelInfo) => void;
  searchQuery: string;
}> = ({ category, onModelClick, searchQuery }) => {
  const [showAll, setShowAll] = useState(false);
  const VISIBLE = 12;

  const filtered = useMemo(() => {
    if (!searchQuery) return category.models;
    const q = searchQuery.toLowerCase();
    return category.models.filter((m) =>
      m.key.toLowerCase().includes(q) ||
      m.litellm_provider.toLowerCase().includes(q) ||
      getProviderDisplayName(m.litellm_provider).toLowerCase().includes(q)
    );
  }, [category.models, searchQuery]);

  if (filtered.length === 0) return null;

  const visible = showAll ? filtered : filtered.slice(0, VISIBLE);

  return (
    <div style={{ marginBottom: 40 }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 4 }}>
        <h2 style={{ fontSize: 18, fontWeight: 400, color: "#202124", margin: 0 }}>{category.label}</h2>
        {filtered.length > VISIBLE && (
          <span onClick={() => setShowAll(!showAll)} style={{
            fontSize: 14, color: "#1a73e8", cursor: "pointer",
            display: "inline-flex", alignItems: "center", gap: 4,
          }}>
            {showAll ? "Show less" : <><ArrowRightOutlined style={{ fontSize: 12 }} /> Show all ({filtered.length})</>}
          </span>
        )}
      </div>
      <p style={{ fontSize: 13, color: "#5f6368", margin: "4px 0 16px 0" }}>{category.description}</p>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 12 }}>
        {visible.map((m) => (
          <ModelCard key={m.key} model={m} onClick={() => onModelClick(m)} />
        ))}
      </div>
    </div>
  );
};

/* ── Sort controls ── */
const SortControls: React.FC<{
  value: "trending" | "latest";
  onChange: (v: "trending" | "latest") => void;
}> = ({ value, onChange }) => (
  <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 24 }}>
    <span style={{ fontSize: 13, color: "#5f6368" }}>Sort by:</span>
    <span onClick={() => onChange("trending")} style={{
      fontSize: 13, fontWeight: value === "trending" ? 600 : 400,
      color: value === "trending" ? "#202124" : "#1a73e8",
      cursor: "pointer", textDecoration: value === "trending" ? "none" : "underline",
    }}>Trending</span>
    <span onClick={() => onChange("latest")} style={{
      fontSize: 13, fontWeight: value === "latest" ? 600 : 400,
      color: value === "latest" ? "#202124" : "#1a73e8",
      cursor: "pointer", textDecoration: value === "latest" ? "none" : "underline",
    }}>Latest</span>
  </div>
);

/* ── Main Component ── */
const ModelGarden: React.FC<ModelGardenProps> = ({ modelCostMap, isLoading }) => {
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedProvider, setSelectedProvider] = useState<ProviderGroup | null>(null);
  const [sortBy, setSortBy] = useState<"trending" | "latest">("trending");

  const providerGroups = useMemo(() => {
    if (!modelCostMap) return [];
    return parseModelCostMap(modelCostMap);
  }, [modelCostMap]);

  const whatsNew = useMemo(() => {
    if (!modelCostMap) return [];
    return detectWhatsNew(modelCostMap);
  }, [modelCostMap]);

  const categories = useMemo(() => {
    if (!modelCostMap) return [];
    return buildModelCategories(modelCostMap);
  }, [modelCostMap]);

  const handleModelClick = (model: ModelInfo) => {
    const group = providerGroups.find((g) => g.provider === model.litellm_provider);
    if (group) setSelectedProvider(group);
  };

  if (selectedProvider) {
    return <ModelGardenDetailView group={selectedProvider} onBack={() => setSelectedProvider(null)} />;
  }

  if (isLoading) {
    return (
      <div style={{ textAlign: "center", padding: 80 }}>
        <Spin size="large" />
        <p style={{ marginTop: 16, color: "#6b7280" }}>Loading model catalog...</p>
      </div>
    );
  }

  const totalModels = providerGroups.reduce((sum, g) => sum + g.modelCount, 0);

  return (
    <div>
      {/* Search */}
      <div style={{ marginBottom: 24 }}>
        <Input size="large" placeholder="Search models, providers, or capabilities..."
          prefix={<SearchOutlined style={{ color: "#9ca3af" }} />}
          value={searchQuery} onChange={(e) => setSearchQuery(e.target.value)}
          style={{ borderRadius: 8, maxWidth: 600 }} />
      </div>

      <p style={{ fontSize: 14, color: "#3c4043", margin: "0 0 24px 0" }}>
        Browse, customize, and deploy machine learning models with <strong>Model Garden</strong>.
        {" "}Choose from {totalModels.toLocaleString()} models across {providerGroups.length} providers.
      </p>

      {/* What's New */}
      {!searchQuery && <WhatsNewSection items={whatsNew} />}

      {/* All Providers carousel */}
      {!searchQuery && (
        <ProviderCarousel providers={providerGroups} onSelect={setSelectedProvider} />
      )}

      {/* Sort controls */}
      <SortControls value={sortBy} onChange={setSortBy} />

      {/* Categories */}
      {categories.map((cat) => (
        <CategorySection key={cat.key} category={cat} onModelClick={handleModelClick} searchQuery={searchQuery} />
      ))}
    </div>
  );
};

export default ModelGarden;
