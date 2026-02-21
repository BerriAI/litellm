import React, { useState } from "react";
import { Input, Button } from "antd";
import { SearchOutlined, ArrowLeftOutlined, ArrowRightOutlined, CheckCircleFilled } from "@ant-design/icons";
import AddGuardrailForm from "./add_guardrail_form";
import { GUARDRAIL_PRESETS } from "./guardrail_garden_configs";

interface GuardrailCardInfo {
  id: string;
  name: string;
  description: string;
  category: "litellm" | "partner";
  subcategory?: string;
  logo: string;
  tags: string[];
  eval?: {
    f1: number;
    precision: number;
    recall: number;
    testCases: number;
    latency: string;
  };
  providerKey?: string;
}

const ASSET_PREFIX = "../assets/logos/";

const LITELLM_CONTENT_FILTER_CARDS: GuardrailCardInfo[] = [
  {
    id: "cf_denied_financial",
    name: "Denied Financial Advice",
    description: "Detects requests for personalized financial advice, investment recommendations, or financial planning.",
    category: "litellm",
    subcategory: "Content Category",
    logo: `${ASSET_PREFIX}litellm_logo.jpg`,
    tags: ["Content Category", "Topic Blocker"],
    eval: {
      f1: 100.0,
      precision: 100.0,
      recall: 100.0,
      testCases: 207,
      latency: "<0.1ms",
    },
  },
  {
    id: "cf_denied_legal",
    name: "Denied Legal Advice",
    description: "Detects requests for unauthorized legal advice, case analysis, or legal recommendations.",
    category: "litellm",
    subcategory: "Content Category",
    logo: `${ASSET_PREFIX}litellm_logo.jpg`,
    tags: ["Content Category", "Topic Blocker"],
  },
  {
    id: "cf_denied_medical",
    name: "Denied Medical Advice",
    description: "Detects requests for medical diagnosis, treatment recommendations, or health advice.",
    category: "litellm",
    subcategory: "Content Category",
    logo: `${ASSET_PREFIX}litellm_logo.jpg`,
    tags: ["Content Category", "Topic Blocker"],
  },
  {
    id: "cf_harmful_violence",
    name: "Harmful Violence",
    description: "Detects content related to violence, criminal planning, attacks, and violent threats.",
    category: "litellm",
    subcategory: "Content Category",
    logo: `${ASSET_PREFIX}litellm_logo.jpg`,
    tags: ["Content Category", "Safety"],
  },
  {
    id: "cf_harmful_self_harm",
    name: "Harmful Self-Harm",
    description: "Detects content related to self-harm, suicide, and dangerous self-destructive behavior.",
    category: "litellm",
    subcategory: "Content Category",
    logo: `${ASSET_PREFIX}litellm_logo.jpg`,
    tags: ["Content Category", "Safety"],
  },
  {
    id: "cf_harmful_child_safety",
    name: "Harmful Child Safety",
    description: "Detects content that could endanger child safety or exploit minors.",
    category: "litellm",
    subcategory: "Content Category",
    logo: `${ASSET_PREFIX}litellm_logo.jpg`,
    tags: ["Content Category", "Safety"],
  },
  {
    id: "cf_harmful_illegal_weapons",
    name: "Harmful Illegal Weapons",
    description: "Detects content related to illegal weapons manufacturing, distribution, or acquisition.",
    category: "litellm",
    subcategory: "Content Category",
    logo: `${ASSET_PREFIX}litellm_logo.jpg`,
    tags: ["Content Category", "Safety"],
  },
  {
    id: "cf_bias_gender",
    name: "Bias: Gender",
    description: "Detects gender-based discrimination, stereotypes, and biased language.",
    category: "litellm",
    subcategory: "Content Category",
    logo: `${ASSET_PREFIX}litellm_logo.jpg`,
    tags: ["Content Category", "Bias"],
  },
  {
    id: "cf_bias_racial",
    name: "Bias: Racial",
    description: "Detects racial discrimination, stereotypes, and racially biased content.",
    category: "litellm",
    subcategory: "Content Category",
    logo: `${ASSET_PREFIX}litellm_logo.jpg`,
    tags: ["Content Category", "Bias"],
  },
  {
    id: "cf_bias_religious",
    name: "Bias: Religious",
    description: "Detects religious discrimination, intolerance, and religiously biased content.",
    category: "litellm",
    subcategory: "Content Category",
    logo: `${ASSET_PREFIX}litellm_logo.jpg`,
    tags: ["Content Category", "Bias"],
  },
  {
    id: "cf_bias_sexual_orientation",
    name: "Bias: Sexual Orientation",
    description: "Detects discrimination based on sexual orientation and related biased content.",
    category: "litellm",
    subcategory: "Content Category",
    logo: `${ASSET_PREFIX}litellm_logo.jpg`,
    tags: ["Content Category", "Bias"],
  },
  {
    id: "cf_prompt_injection_jailbreak",
    name: "Prompt Injection: Jailbreak",
    description: "Detects jailbreak attempts designed to bypass AI safety guidelines and restrictions.",
    category: "litellm",
    subcategory: "Content Category",
    logo: `${ASSET_PREFIX}litellm_logo.jpg`,
    tags: ["Content Category", "Prompt Injection"],
  },
  {
    id: "cf_prompt_injection_data_exfil",
    name: "Prompt Injection: Data Exfiltration",
    description: "Detects attempts to extract sensitive data through prompt manipulation.",
    category: "litellm",
    subcategory: "Content Category",
    logo: `${ASSET_PREFIX}litellm_logo.jpg`,
    tags: ["Content Category", "Prompt Injection"],
  },
  {
    id: "cf_prompt_injection_sql",
    name: "Prompt Injection: SQL",
    description: "Detects SQL injection attempts embedded in prompts.",
    category: "litellm",
    subcategory: "Content Category",
    logo: `${ASSET_PREFIX}litellm_logo.jpg`,
    tags: ["Content Category", "Prompt Injection"],
  },
  {
    id: "cf_prompt_injection_malicious_code",
    name: "Prompt Injection: Malicious Code",
    description: "Detects attempts to inject malicious code through prompts.",
    category: "litellm",
    subcategory: "Content Category",
    logo: `${ASSET_PREFIX}litellm_logo.jpg`,
    tags: ["Content Category", "Prompt Injection"],
  },
  {
    id: "cf_prompt_injection_system_prompt",
    name: "Prompt Injection: System Prompt",
    description: "Detects attempts to extract or override system prompts.",
    category: "litellm",
    subcategory: "Content Category",
    logo: `${ASSET_PREFIX}litellm_logo.jpg`,
    tags: ["Content Category", "Prompt Injection"],
  },
  {
    id: "cf_denied_insults",
    name: "Insults & Personal Attacks",
    description: "Detects insults, name-calling, and personal attacks directed at the chatbot, staff, or other people.",
    category: "litellm",
    subcategory: "Content Category",
    logo: `${ASSET_PREFIX}litellm_logo.jpg`,
    tags: ["Content Category", "Topic Blocker"],
    eval: {
      f1: 100.0,
      precision: 100.0,
      recall: 100.0,
      testCases: 299,
      latency: "<0.1ms",
    },
  },
  {
    id: "cf_toxic_abuse",
    name: "Toxic & Abusive Language",
    description: "Detects toxic, abusive, and hateful language across multiple languages (EN, AU, DE, ES, FR).",
    category: "litellm",
    subcategory: "Content Category",
    logo: `${ASSET_PREFIX}litellm_logo.jpg`,
    tags: ["Content Category", "Toxicity"],
  },
  {
    id: "cf_patterns",
    name: "Pattern Matching",
    description: "Detect and block sensitive data patterns like SSNs, credit card numbers, API keys, and custom regex patterns.",
    category: "litellm",
    subcategory: "Patterns",
    logo: `${ASSET_PREFIX}litellm_logo.jpg`,
    tags: ["PII", "Regex", "Data Protection"],
  },
  {
    id: "cf_keywords",
    name: "Keyword Blocking",
    description: "Block or mask content containing specific keywords or phrases. Upload custom word lists or add individual terms.",
    category: "litellm",
    subcategory: "Keywords",
    logo: `${ASSET_PREFIX}litellm_logo.jpg`,
    tags: ["Keywords", "Blocklist"],
  },
];

const PARTNER_GUARDRAIL_CARDS: GuardrailCardInfo[] = [
  {
    id: "presidio",
    name: "Presidio PII",
    description: "Microsoft Presidio for PII detection and anonymization. Supports 30+ entity types with configurable actions.",
    category: "partner",
    logo: `${ASSET_PREFIX}presidio.png`,
    tags: ["PII", "Microsoft"],
    providerKey: "PresidioPII",
  },
  {
    id: "bedrock",
    name: "Bedrock Guardrail",
    description: "AWS Bedrock Guardrails for content filtering, topic avoidance, and sensitive information detection.",
    category: "partner",
    logo: `${ASSET_PREFIX}bedrock.svg`,
    tags: ["AWS", "Content Safety"],
    providerKey: "Bedrock",
  },
  {
    id: "lakera",
    name: "Lakera",
    description: "AI security platform protecting against prompt injections, data leakage, and harmful content.",
    category: "partner",
    logo: `${ASSET_PREFIX}lakeraai.jpeg`,
    tags: ["Security", "Prompt Injection"],
    providerKey: "Lakera",
  },
  {
    id: "openai_moderation",
    name: "OpenAI Moderation",
    description: "OpenAI's content moderation API for detecting harmful content across multiple categories.",
    category: "partner",
    logo: `${ASSET_PREFIX}openai_small.svg`,
    tags: ["Content Moderation", "OpenAI"],
  },
  {
    id: "google_model_armor",
    name: "Google Cloud Model Armor",
    description: "Google Cloud's model protection service for safe and responsible AI deployments.",
    category: "partner",
    logo: `${ASSET_PREFIX}google.svg`,
    tags: ["Google Cloud", "Safety"],
  },
  {
    id: "guardrails_ai",
    name: "Guardrails AI",
    description: "Open-source framework for adding structural, type, and quality guarantees to LLM outputs.",
    category: "partner",
    logo: `${ASSET_PREFIX}guardrails_ai.jpeg`,
    tags: ["Open Source", "Validation"],
  },
  {
    id: "zscaler",
    name: "Zscaler AI Guard",
    description: "Enterprise AI security from Zscaler for monitoring and protecting AI/ML workloads.",
    category: "partner",
    logo: `${ASSET_PREFIX}zscaler.svg`,
    tags: ["Enterprise", "Security"],
  },
  {
    id: "panw",
    name: "PANW Prisma AIRS",
    description: "Palo Alto Networks Prisma AI Runtime Security for securing AI applications in production.",
    category: "partner",
    logo: `${ASSET_PREFIX}palo_alto_networks.jpeg`,
    tags: ["Enterprise", "Security"],
  },
  {
    id: "noma",
    name: "Noma Security",
    description: "AI security platform for detecting and preventing AI-specific threats and vulnerabilities.",
    category: "partner",
    logo: `${ASSET_PREFIX}noma_security.png`,
    tags: ["Security", "Threat Detection"],
  },
  {
    id: "aporia",
    name: "Aporia AI",
    description: "Real-time AI guardrails for hallucination detection, topic control, and policy enforcement.",
    category: "partner",
    logo: `${ASSET_PREFIX}aporia.png`,
    tags: ["Hallucination", "Policy"],
  },
  {
    id: "aim",
    name: "AIM Guardrail",
    description: "AIM Security guardrails for comprehensive AI threat detection and mitigation.",
    category: "partner",
    logo: `${ASSET_PREFIX}aim_security.jpeg`,
    tags: ["Security", "Threat Detection"],
  },
  {
    id: "prompt_security",
    name: "Prompt Security",
    description: "Protect against prompt injection attacks, data leakage, and other LLM security threats.",
    category: "partner",
    logo: `${ASSET_PREFIX}prompt_security.png`,
    tags: ["Prompt Injection", "Security"],
  },
  {
    id: "lasso",
    name: "Lasso Guardrail",
    description: "Content moderation and safety guardrails for responsible AI deployments.",
    category: "partner",
    logo: `${ASSET_PREFIX}lasso.png`,
    tags: ["Content Moderation"],
  },
  {
    id: "pangea",
    name: "Pangea Guardrail",
    description: "Pangea's AI guardrails for secure, compliant, and trustworthy AI applications.",
    category: "partner",
    logo: `${ASSET_PREFIX}pangea.png`,
    tags: ["Compliance", "Security"],
  },
  {
    id: "enkryptai",
    name: "EnkryptAI",
    description: "AI security and governance platform for enterprise AI safety and compliance.",
    category: "partner",
    logo: `${ASSET_PREFIX}enkrypt_ai.avif`,
    tags: ["Enterprise", "Governance"],
  },
  {
    id: "javelin",
    name: "Javelin Guardrails",
    description: "AI gateway with built-in guardrails for secure and compliant AI operations.",
    category: "partner",
    logo: `${ASSET_PREFIX}javelin.png`,
    tags: ["Gateway", "Security"],
  },
  {
    id: "pillar",
    name: "Pillar Guardrail",
    description: "AI safety platform for monitoring, testing, and securing AI systems.",
    category: "partner",
    logo: `${ASSET_PREFIX}pillar.jpeg`,
    tags: ["Monitoring", "Safety"],
  },
];

const ALL_CARDS = [...LITELLM_CONTENT_FILTER_CARDS, ...PARTNER_GUARDRAIL_CARDS];

interface GuardrailGardenProps {
  accessToken: string | null;
  onGuardrailCreated: () => void;
}

const GuardrailGarden: React.FC<GuardrailGardenProps> = ({ accessToken, onGuardrailCreated }) => {
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
          placeholder="Search guardrails"
          prefix={<SearchOutlined style={{ color: "#9ca3af" }} />}
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          style={{ borderRadius: 8 }}
        />
      </div>

      {/* LiteLLM Content Filter Section */}
      <div style={{ marginBottom: 40 }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 4 }}>
          <h2 style={{ fontSize: 20, fontWeight: 600, color: "#111827", margin: 0 }}>LiteLLM Content Filter</h2>
          <span
            style={{ display: "inline-flex", alignItems: "center", gap: 6, fontSize: 14, color: "#1a73e8", cursor: "pointer" }}
            onClick={() => setShowAllLitellm(!showAllLitellm)}
          >
            {showAllLitellm ? (
              <>Show less</>
            ) : (
              <>
                <ArrowRightOutlined style={{ fontSize: 12 }} />
                {`Show all (${litellmCards.length})`}
              </>
            )}
          </span>
        </div>
        <p style={{ fontSize: 13, color: "#6b7280", margin: "4px 0 20px 0" }}>
          Built-in guardrails powered by LiteLLM. Zero latency, no external dependencies, no additional cost.
        </p>
        <div style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fill, minmax(220px, 1fr))",
          gap: 16,
        }}>
          {(showAllLitellm ? litellmCards : litellmCards.slice(0, CARDS_PER_ROW * VISIBLE_ROWS)).map((card) => (
            <GuardrailCard key={card.id} card={card} onClick={() => setSelectedCard(card)} />
          ))}
        </div>
      </div>

      {/* Partner Guardrails Section */}
      <div style={{ marginBottom: 40 }}>
        <h2 style={{ fontSize: 20, fontWeight: 600, color: "#111827", margin: "0 0 4px 0" }}>Partner Guardrails</h2>
        <p style={{ fontSize: 13, color: "#6b7280", margin: "4px 0 20px 0" }}>
          Third-party guardrail integrations from leading AI security providers.
        </p>
        <div style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fill, minmax(220px, 1fr))",
          gap: 16,
        }}>
          {partnerCards.map((card) => (
            <GuardrailCard key={card.id} card={card} onClick={() => setSelectedCard(card)} />
          ))}
        </div>
      </div>
    </div>
  );
};

/* ── Card (Vertex-style) ─────────────────────────────────── */
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
        <img
          src={card.logo}
          alt=""
          style={{ width: 28, height: 28, borderRadius: 6, objectFit: "contain", flexShrink: 0 }}
          onError={(e) => { (e.target as HTMLImageElement).style.display = "none"; }}
        />
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

/* ── Detail view (Vertex-style) ──────────────────────────── */
interface GuardrailDetailViewProps {
  card: GuardrailCardInfo;
  onBack: () => void;
  accessToken: string | null;
  onGuardrailCreated: () => void;
}

const GuardrailDetailView: React.FC<GuardrailDetailViewProps> = ({
  card,
  onBack,
  accessToken,
  onGuardrailCreated,
}) => {
  const [isAddFormVisible, setIsAddFormVisible] = useState(false);
  const [activeTab, setActiveTab] = useState("overview");

  const detailRows = [
    { property: "Provider", value: card.category === "litellm" ? "LiteLLM Content Filter" : "Partner Guardrail" },
    ...(card.subcategory ? [{ property: "Subcategory", value: card.subcategory }] : []),
    ...(card.category === "litellm" ? [{ property: "Cost", value: "$0 / request" }] : []),
    ...(card.category === "litellm" ? [{ property: "External Dependencies", value: "None" }] : []),
    ...(card.category === "litellm" ? [{ property: "Latency", value: card.eval?.latency || "<1ms" }] : []),
  ];

  const evalRows = card.eval
    ? [
        { metric: "Precision", value: `${card.eval.precision}%` },
        { metric: "Recall", value: `${card.eval.recall}%` },
        { metric: "F1 Score", value: `${card.eval.f1}%` },
        { metric: "Test Cases", value: String(card.eval.testCases) },
        { metric: "False Positives", value: "0" },
        { metric: "False Negatives", value: "0" },
        { metric: "Latency (p50)", value: card.eval.latency },
      ]
    : [];

  const tabs = [
    { key: "overview", label: "Overview" },
    ...(card.eval ? [{ key: "eval", label: "Eval Results" }] : []),
  ];

  return (
    <div style={{ maxWidth: 960, margin: "0 auto" }}>
      {/* Back link */}
      <div
        onClick={onBack}
        style={{
          display: "inline-flex",
          alignItems: "center",
          gap: 6,
          color: "#5f6368",
          cursor: "pointer",
          fontSize: 14,
          marginBottom: 24,
        }}
      >
        <ArrowLeftOutlined style={{ fontSize: 11 }} />
        <span>{card.name}</span>
      </div>

      {/* ── Header block (Vertex-style) ── */}
      <div style={{ display: "flex", alignItems: "center", gap: 16, marginBottom: 8 }}>
        <img
          src={card.logo}
          alt=""
          style={{ width: 40, height: 40, borderRadius: 8, objectFit: "contain" }}
          onError={(e) => { (e.target as HTMLImageElement).style.display = "none"; }}
        />
        <h1 style={{ fontSize: 28, fontWeight: 400, color: "#202124", margin: 0, lineHeight: 1.2 }}>
          {card.name}
        </h1>
      </div>

      <p style={{ fontSize: 14, color: "#5f6368", margin: "0 0 20px 0", lineHeight: 1.6 }}>
        {card.description}
      </p>

      {/* Action buttons — outlined style like Vertex */}
      <div style={{ display: "flex", gap: 10, marginBottom: 32 }}>
        <Button
          onClick={() => setIsAddFormVisible(true)}
          style={{
            borderRadius: 20,
            padding: "4px 20px",
            height: 36,
            borderColor: "#dadce0",
            color: "#1a73e8",
            fontWeight: 500,
            fontSize: 14,
          }}
        >
          Create Guardrail
        </Button>
      </div>

      {/* ── Tab bar ──────────────────────────────────── */}
      <div style={{ borderBottom: "1px solid #dadce0", marginBottom: 28 }}>
        <div style={{ display: "flex", gap: 0 }}>
          {tabs.map((tab) => (
            <div
              key={tab.key}
              onClick={() => setActiveTab(tab.key)}
              style={{
                padding: "12px 20px",
                fontSize: 14,
                color: activeTab === tab.key ? "#1a73e8" : "#5f6368",
                borderBottom: activeTab === tab.key ? "3px solid #1a73e8" : "3px solid transparent",
                cursor: "pointer",
                fontWeight: activeTab === tab.key ? 500 : 400,
                marginBottom: -1,
              }}
            >
              {tab.label}
            </div>
          ))}
        </div>
      </div>

      {/* ── Tab content ──────────────────────────────── */}
      {activeTab === "overview" && (
        <div style={{ display: "flex", gap: 64 }}>
          {/* Left column — overview + details table */}
          <div style={{ flex: 1, minWidth: 0 }}>
            <h2 style={{ fontSize: 18, fontWeight: 400, color: "#202124", margin: "0 0 12px 0" }}>Overview</h2>
            <p style={{ fontSize: 14, color: "#3c4043", lineHeight: 1.7, margin: "0 0 32px 0" }}>
              {card.description}
            </p>

            <h2 style={{ fontSize: 18, fontWeight: 400, color: "#202124", margin: "0 0 4px 0" }}>Guardrail Details</h2>
            <p style={{ fontSize: 13, color: "#5f6368", margin: "0 0 16px 0" }}>Details are as follows</p>

            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 14 }}>
              <thead>
                <tr style={{ borderBottom: "1px solid #dadce0" }}>
                  <th style={{ textAlign: "left", padding: "12px 0", color: "#5f6368", fontWeight: 500, width: 200 }}>Property</th>
                  <th style={{ textAlign: "left", padding: "12px 0", color: "#5f6368", fontWeight: 500 }}>{card.name}</th>
                </tr>
              </thead>
              <tbody>
                {detailRows.map((row, i) => (
                  <tr key={i} style={{ borderBottom: "1px solid #f1f3f4" }}>
                    <td style={{ padding: "12px 0", color: "#3c4043" }}>{row.property}</td>
                    <td style={{ padding: "12px 0", color: "#202124" }}>{row.value}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Right column — metadata sidebar like Vertex */}
          <div style={{ width: 240, flexShrink: 0 }}>
            {/* Guardrail ID */}
            <div style={{ marginBottom: 28 }}>
              <div style={{ fontSize: 12, color: "#5f6368", marginBottom: 4 }}>Guardrail ID</div>
              <div style={{ fontSize: 13, color: "#202124", wordBreak: "break-all" }}>
                litellm/{card.id}
              </div>
            </div>

            {/* Type */}
            <div style={{ marginBottom: 28 }}>
              <div style={{ fontSize: 12, color: "#5f6368", marginBottom: 4 }}>Type</div>
              <div style={{ fontSize: 13, color: "#202124" }}>
                {card.category === "litellm" ? "Content Filter" : "Partner"}
              </div>
            </div>

            {/* Tags — pill style like Vertex */}
            {card.tags.length > 0 && (
              <div style={{ marginBottom: 28 }}>
                <div style={{ fontSize: 12, color: "#5f6368", marginBottom: 8 }}>Tags</div>
                <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                  {card.tags.map((tag) => (
                    <span
                      key={tag}
                      style={{
                        fontSize: 12,
                        padding: "4px 12px",
                        borderRadius: 16,
                        border: "1px solid #dadce0",
                        color: "#3c4043",
                        backgroundColor: "#fff",
                      }}
                    >
                      {tag}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {activeTab === "eval" && (
        <div>
          <h2 style={{ fontSize: 18, fontWeight: 400, color: "#202124", margin: "0 0 16px 0" }}>Eval Results</h2>
          <table style={{ width: "100%", maxWidth: 560, borderCollapse: "collapse", fontSize: 14 }}>
            <thead>
              <tr style={{ backgroundColor: "#f8f9fa", borderBottom: "1px solid #dadce0" }}>
                <th style={{ textAlign: "left", padding: "12px 16px", color: "#5f6368", fontWeight: 500 }}>Metric</th>
                <th style={{ textAlign: "left", padding: "12px 16px", color: "#5f6368", fontWeight: 500 }}>Value</th>
              </tr>
            </thead>
            <tbody>
              {evalRows.map((row, i) => (
                <tr key={i} style={{ borderBottom: "1px solid #f1f3f4" }}>
                  <td style={{ padding: "12px 16px", color: "#3c4043" }}>{row.metric}</td>
                  <td style={{ padding: "12px 16px", color: "#202124", fontWeight: 500 }}>{row.value}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <AddGuardrailForm
        visible={isAddFormVisible}
        onClose={() => setIsAddFormVisible(false)}
        accessToken={accessToken}
        onSuccess={() => {
          setIsAddFormVisible(false);
          onGuardrailCreated();
        }}
        preset={GUARDRAIL_PRESETS[card.id]}
      />
    </div>
  );
};

export default GuardrailGarden;
