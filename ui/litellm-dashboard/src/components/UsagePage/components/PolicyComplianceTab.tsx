import React, { useMemo, useState } from "react";
import {
  BarChart,
  Card,
  Col,
  DateRangePickerValue,
  Grid,
  Text,
  Title,
} from "@tremor/react";
import { Table, Segmented, Drawer } from "antd";
import type { ColumnsType } from "antd/es/table";
import AdvancedDatePicker from "../../shared/advanced_date_picker";

// ---- Mock Data ----

const MOCK_KPI = {
  totalRequests: 1247,
  euAiActViolations: 23,
  gdprViolations: 41,
  mcpUnregisteredCalls: 8,
  compliantRequests: 1175,
};

// Daily totals must sum to: Compliant=1175, EU AI Act=23, GDPR=41, MCP=8 → Total=1247
const MOCK_DAILY_VIOLATIONS = [
  { date: "Feb 10", Compliant: 148, "EU AI Act": 2, GDPR: 5, "MCP Unregistered": 1 },
  { date: "Feb 11", Compliant: 162, "EU AI Act": 3, GDPR: 6, "MCP Unregistered": 1 },
  { date: "Feb 12", Compliant: 155, "EU AI Act": 4, GDPR: 7, "MCP Unregistered": 2 },
  { date: "Feb 13", Compliant: 170, "EU AI Act": 5, GDPR: 7, "MCP Unregistered": 1 },
  { date: "Feb 14", Compliant: 160, "EU AI Act": 3, GDPR: 5, "MCP Unregistered": 1 },
  { date: "Feb 15", Compliant: 175, "EU AI Act": 2, GDPR: 4, "MCP Unregistered": 1 },
  { date: "Feb 16", Compliant: 140, "EU AI Act": 3, GDPR: 5, "MCP Unregistered": 1 },
  { date: "Feb 17", Compliant: 65, "EU AI Act": 1, GDPR: 2, "MCP Unregistered": 0 },
];

interface TeamViolation {
  key: string;
  team: string;
  euAiAct: number;
  gdpr: number;
  mcpUnregistered: number;
  risk: "HIGH" | "MED" | "LOW";
}

// Team violations must sum to: EU AI Act=23, GDPR=41, MCP=8
const MOCK_TEAM_VIOLATIONS: TeamViolation[] = [
  { key: "1", team: "HR Automation Bot", euAiAct: 12, gdpr: 18, mcpUnregistered: 0, risk: "HIGH" },
  { key: "2", team: "Internal Doc Search", euAiAct: 0, gdpr: 10, mcpUnregistered: 5, risk: "HIGH" },
  { key: "3", team: "Contract Analyzer", euAiAct: 6, gdpr: 9, mcpUnregistered: 3, risk: "MED" },
  { key: "4", team: "Customer Support", euAiAct: 3, gdpr: 4, mcpUnregistered: 0, risk: "LOW" },
  { key: "5", team: "Platform Chatbot", euAiAct: 2, gdpr: 0, mcpUnregistered: 0, risk: "LOW" },
];

// Regulation articles must sum to: EU AI Act (5+9+12)=23, GDPR (32+38)=41
const MOCK_REGULATION_ARTICLES = [
  { article: "Art. 32 GDPR (Data Protection)", count: 28 },
  { article: "Art. 5 (Prohibited Practices)", count: 12 },
  { article: "Art. 38 GDPR (Audit Records)", count: 13 },
  { article: "Art. 9 (Risk Management)", count: 6 },
  { article: "Art. 12 (Transparency)", count: 5 },
];

// Request type violations must sum to total violations: 23+41+8=72
const MOCK_REQUEST_TYPE_VIOLATIONS = [
  { type: "LLM Calls", violations: 52 },
  { type: "MCP Tool Calls", violations: 12 },
  { type: "Agent Calls", violations: 8 },
];

interface DrillDownLog {
  key: string;
  timestamp: string;
  requestId: string;
  regulation: "EU AI Act" | "GDPR" | "MCP Unregistered";
  article: string;
  severity: "critical" | "high" | "medium";
  model: string;
  virtualKey: string;
  requestType: "LLM Call" | "MCP Tool Call" | "Agent Call";
  inputSnippet: string;
  violationReason: string;
  recommendation: string;
}

const MOCK_DRILL_DOWN: Record<string, DrillDownLog[]> = {
  "HR Automation Bot": [
    {
      key: "1", timestamp: "2026-02-17 09:12:34", requestId: "req_8f3a1b2c",
      regulation: "EU AI Act", article: "Art. 5 (Prohibited Practices)", severity: "critical",
      model: "gpt-4o", virtualKey: "sk-hr-bot-prod", requestType: "LLM Call",
      inputSnippet: "Based on the employee's performance score of 2.1/10 and attendance record, generate a termination letter and notify HR to proceed with dismissal...",
      violationReason: "Automated decision-making on employment termination without mandatory human oversight. Art. 5(1)(c) prohibits AI systems that evaluate or classify persons based on social behavior leading to detrimental treatment.",
      recommendation: "Add human-in-the-loop approval before any employment decisions. Route output to HR manager for review before action.",
    },
    {
      key: "2", timestamp: "2026-02-17 08:45:12", requestId: "req_2d4e6f8a",
      regulation: "GDPR", article: "Art. 32 (Data Protection)", severity: "critical",
      model: "gpt-4o", virtualKey: "sk-hr-bot-prod", requestType: "LLM Call",
      inputSnippet: "Employee record: Name: John Smith, SSN: 412-55-8901, DOB: 1985-03-14, Medical leave history: 3 instances of mental health leave in 2025...",
      violationReason: "Unencrypted PII (SSN, date of birth) and special category data (health records) sent to external LLM provider without data protection measures.",
      recommendation: "Mask or tokenize PII before sending to LLM. Use litellm guardrails to detect and redact sensitive fields (pii_masking). Never send health data to external providers.",
    },
    {
      key: "3", timestamp: "2026-02-16 14:22:08", requestId: "req_9c1d3e5f",
      regulation: "GDPR", article: "Art. 32 (Data Protection)", severity: "high",
      model: "claude-3-5-sonnet", virtualKey: "sk-hr-bot-prod", requestType: "LLM Call",
      inputSnippet: "Analyser le dossier de Marie Dupont: adresse 12 rue de la Paix Paris, numero secu 2 85 03 75 108 042 15, evaluations de performance 2024-2025...",
      violationReason: "French national ID number (numero de securite sociale) and home address transmitted to LLM without consent or encryption.",
      recommendation: "Enable PII guardrail for French ID patterns. Require explicit consent before processing employee evaluations with AI.",
    },
    {
      key: "4", timestamp: "2026-02-16 11:03:55", requestId: "req_4b6c8d0e",
      regulation: "EU AI Act", article: "Art. 5 (Prohibited Practices)", severity: "critical",
      model: "gpt-4o", virtualKey: "sk-hr-bot-prod", requestType: "LLM Call",
      inputSnippet: "Rank all employees in the engineering department by: productivity score, peer review sentiment, Slack activity metrics, badge-in frequency. Flag bottom 10% for performance improvement plan...",
      violationReason: "Social scoring of employees using behavioral surveillance data (Slack activity, badge-in frequency). This constitutes prohibited social scoring under Art. 5(1)(c).",
      recommendation: "Remove behavioral surveillance inputs. Performance reviews must use only job-relevant, transparent criteria with employee awareness.",
    },
    {
      key: "5", timestamp: "2026-02-15 16:47:21", requestId: "req_7a9b1c3d",
      regulation: "GDPR", article: "Art. 32 (Data Protection)", severity: "high",
      model: "gpt-4o", virtualKey: "sk-hr-bot-prod", requestType: "LLM Call",
      inputSnippet: "Summarize sick leave patterns for the following employees and flag anyone with >5 days mental health leave: [list of 47 employees with full medical records]...",
      violationReason: "Bulk processing of health data (special category under Art. 9 GDPR) without explicit consent or legitimate basis. Data sent to US-based provider without adequate safeguards.",
      recommendation: "Health data processing requires explicit employee consent per Art. 9(2)(a). Aggregate and anonymize before any AI analysis. Consider EU-hosted model.",
    },
  ],
  "Internal Doc Search": [
    {
      key: "1", timestamp: "2026-02-17 10:05:18", requestId: "req_1e2f3a4b",
      regulation: "GDPR", article: "Art. 32 (Data Protection)", severity: "high",
      model: "text-embedding-3-small", virtualKey: "sk-docsearch-prod", requestType: "LLM Call",
      inputSnippet: "Search query: 'Find all contracts mentioning employee salary bands for Sarah Chen, Michael Rodriguez, and compensation packages above 200k'...",
      violationReason: "Search query retrieves and exposes individual salary data (personal data) without access controls or legitimate business need verification.",
      recommendation: "Add role-based access controls to document search. Salary data queries should require manager-level permissions and audit logging.",
    },
    {
      key: "2", timestamp: "2026-02-16 09:33:41", requestId: "req_5c6d7e8f",
      regulation: "MCP Unregistered", article: "MCP Unregistered Server", severity: "medium",
      model: "gpt-4o", virtualKey: "sk-docsearch-prod", requestType: "MCP Tool Call",
      inputSnippet: "Tool call to 'internal-search-v2' server at endpoint https://search-staging.internal:8443/query — server not found in MCP registry...",
      violationReason: "MCP tool call routed to unregistered server 'internal-search-v2'. This server is not in the approved MCP registry and has not been security-reviewed.",
      recommendation: "Register 'internal-search-v2' in the MCP server registry via Settings > MCP Servers. Ensure security review is completed before production use.",
    },
    {
      key: "3", timestamp: "2026-02-15 15:22:09", requestId: "req_9a0b1c2d",
      regulation: "GDPR", article: "Art. 32 (Data Protection)", severity: "high",
      model: "text-embedding-3-small", virtualKey: "sk-docsearch-prod", requestType: "LLM Call",
      inputSnippet: "Recherche: 'dossiers medicaux employes site Lyon, certificats arret maladie 2025, notes medecin du travail'...",
      violationReason: "Search query targets medical records (special category data). Embedding model processes sensitive health information without adequate protection.",
      recommendation: "Exclude medical/health document collections from general search index. Create separate, access-controlled index with explicit consent requirements.",
    },
  ],
  "Contract Analyzer": [
    {
      key: "1", timestamp: "2026-02-17 07:55:02", requestId: "req_3d4e5f6a",
      regulation: "EU AI Act", article: "Art. 9 (Risk Management)", severity: "high",
      model: "claude-3-5-sonnet", virtualKey: "sk-contracts-prod", requestType: "LLM Call",
      inputSnippet: "Analyze this $4.2M vendor contract and recommend whether to approve or reject. Key terms: liability cap, SLA penalties, data processing addendum. Auto-approve if risk score < 0.3...",
      violationReason: "High-risk AI decision (contract approval >$1M) without mandatory risk assessment documentation. Art. 9 requires documented risk management for high-value automated decisions.",
      recommendation: "Contracts above threshold must go through documented risk assessment. Add human approval step for AI-recommended contract decisions above $1M.",
    },
    {
      key: "2", timestamp: "2026-02-16 13:18:45", requestId: "req_7b8c9d0e",
      regulation: "GDPR", article: "Art. 38 (Audit Records)", severity: "medium",
      model: "gpt-4o", virtualKey: "sk-contracts-prod", requestType: "LLM Call",
      inputSnippet: "Extract all personal data subjects mentioned in the attached data processing agreement. List names, roles, and data categories processed...",
      violationReason: "Contract analysis extracting personal data without maintaining required audit records. Art. 38 requires DPO notification and logging for data subject identification activities.",
      recommendation: "Enable detailed audit logging for all contract analysis requests involving personal data. Notify DPO when data subject identification is performed.",
    },
  ],
  "Customer Support": [
    {
      key: "1", timestamp: "2026-02-14 11:22:33", requestId: "req_1f2a3b4c",
      regulation: "EU AI Act", article: "Art. 12 (Transparency)", severity: "medium",
      model: "gpt-4o-mini", virtualKey: "sk-support-prod", requestType: "LLM Call",
      inputSnippet: "Customer asked: 'Am I speaking with a real person?' System prompt instructs: 'You are a helpful customer service representative named Alex. Never reveal you are an AI.'...",
      violationReason: "AI system instructed to conceal its nature when directly asked by user. Art. 12 requires AI systems to be transparent about their non-human nature.",
      recommendation: "Update system prompt to disclose AI nature when asked. Add standard disclosure: 'I'm an AI assistant powered by [company]. I can connect you with a human agent.'",
    },
    {
      key: "2", timestamp: "2026-02-13 09:44:17", requestId: "req_5d6e7f8a",
      regulation: "EU AI Act", article: "Art. 12 (Transparency)", severity: "medium",
      model: "gpt-4o-mini", virtualKey: "sk-support-prod", requestType: "LLM Call",
      inputSnippet: "Le client demande: 'Est-ce que je parle a un humain ou a un robot?' Instruction systeme: 'Repondre comme un agent humain, ne pas mentionner l'IA'...",
      violationReason: "Same transparency violation in French-language support channel. Customer explicitly asked if speaking to AI and system is instructed to deny it.",
      recommendation: "Apply the same transparency fix across all language channels. System prompt must allow AI self-identification in all supported languages.",
    },
  ],
  "Platform Chatbot": [
    {
      key: "1", timestamp: "2026-02-12 16:08:52", requestId: "req_9b0c1d2e",
      regulation: "EU AI Act", article: "Art. 12 (Transparency)", severity: "medium",
      model: "gpt-4o-mini", virtualKey: "sk-chatbot-prod", requestType: "Agent Call",
      inputSnippet: "Chatbot greeting: 'Hi! I'm your personal assistant. How can I help you today?' — no AI disclosure in greeting or system prompt...",
      violationReason: "Public-facing chatbot does not identify itself as an AI system at any point in the interaction. Art. 12 requires clear disclosure before or at the start of interaction.",
      recommendation: "Add AI disclosure to chatbot greeting: 'Hi! I'm an AI assistant for [Platform]. How can I help?' Also add disclosure in the chat widget UI.",
    },
  ],
};

// ---- Component ----

const PolicyComplianceTab: React.FC = () => {
  const initialFromDate = useMemo(() => new Date(Date.now() - 7 * 24 * 60 * 60 * 1000), []);
  const initialToDate = useMemo(() => new Date(), []);
  const [dateValue, setDateValue] = useState<DateRangePickerValue>({
    from: initialFromDate,
    to: initialToDate,
  });
  const [teamPageSize, setTeamPageSize] = useState<number>(5);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [selectedTeam, setSelectedTeam] = useState<string | null>(null);

  const violationCount = (val: number) => {
    if (val === 0) return <span className="text-gray-300 font-normal">{val}</span>;
    if (val > 5) return <span className="text-red-600 font-semibold">{val}</span>;
    return <span className="text-orange-500 font-medium">{val}</span>;
  };

  const riskBadgeStyle = (risk: string): React.CSSProperties => {
    const base: React.CSSProperties = {
      display: "inline-block",
      padding: "1px 8px",
      borderRadius: "10px",
      fontSize: "12px",
      fontWeight: 500,
      lineHeight: "20px",
    };
    switch (risk) {
      case "HIGH": return { ...base, backgroundColor: "#fef2f2", color: "#dc2626" };
      case "MED": return { ...base, backgroundColor: "#fff7ed", color: "#d97706" };
      case "LOW": return { ...base, backgroundColor: "#f0fdf4", color: "#16a34a" };
      default: return { ...base, backgroundColor: "#f3f4f6", color: "#6b7280" };
    }
  };

  const columnHeader = (title: string, sub: string) => (
    <div>
      <div>{title}</div>
      <div style={{ fontSize: "11px", color: "#9ca3af", fontWeight: 400 }}>{sub}</div>
    </div>
  );

  const teamColumns: ColumnsType<TeamViolation> = [
    {
      title: "Team / Use Case",
      dataIndex: "team",
      key: "team",
      render: (text: string) => (
        <a
          onClick={() => { setSelectedTeam(text); setDrawerOpen(true); }}
          className="text-blue-600 hover:text-blue-800 cursor-pointer"
        >
          {text}
        </a>
      ),
    },
    {
      title: columnHeader("EU AI Act", "(violations)"),
      dataIndex: "euAiAct",
      key: "euAiAct",
      render: violationCount,
    },
    {
      title: columnHeader("GDPR", "(violations)"),
      dataIndex: "gdpr",
      key: "gdpr",
      render: violationCount,
    },
    {
      title: columnHeader("MCP Unregistered", "(violations)"),
      dataIndex: "mcpUnregistered",
      key: "mcpUnregistered",
      render: violationCount,
    },
    {
      title: "Risk",
      dataIndex: "risk",
      key: "risk",
      render: (risk: string) => <span style={riskBadgeStyle(risk)}>{risk}</span>,
    },
  ];

  const severityBadge = (severity: string) => {
    const styles: Record<string, React.CSSProperties> = {
      critical: { backgroundColor: "#fef2f2", color: "#dc2626", padding: "1px 8px", borderRadius: "10px", fontSize: "12px", fontWeight: 500 },
      high: { backgroundColor: "#fff7ed", color: "#d97706", padding: "1px 8px", borderRadius: "10px", fontSize: "12px", fontWeight: 500 },
      medium: { backgroundColor: "#fefce8", color: "#a16207", padding: "1px 8px", borderRadius: "10px", fontSize: "12px", fontWeight: 500 },
    };
    return <span style={styles[severity] || styles.medium}>{severity.toUpperCase()}</span>;
  };

  const regulationBadge = (reg: string) => {
    const styles: Record<string, React.CSSProperties> = {
      "EU AI Act": { backgroundColor: "#eef2ff", color: "#4338ca", padding: "1px 8px", borderRadius: "10px", fontSize: "12px", fontWeight: 500 },
      "GDPR": { backgroundColor: "#f0fdf4", color: "#15803d", padding: "1px 8px", borderRadius: "10px", fontSize: "12px", fontWeight: 500 },
      "MCP Unregistered": { backgroundColor: "#fff7ed", color: "#c2410c", padding: "1px 8px", borderRadius: "10px", fontSize: "12px", fontWeight: 500 },
    };
    return <span style={styles[reg] || {}}>{reg}</span>;
  };

  const renderViolationCard = (log: DrillDownLog) => (
    <Card key={log.key} className="mb-3">
      {/* Header row */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          {severityBadge(log.severity)}
          {regulationBadge(log.regulation)}
          <span className="text-xs text-gray-400">{log.article}</span>
        </div>
        <span className="text-xs text-gray-400">{log.timestamp}</span>
      </div>

      {/* Meta row */}
      <div className="flex gap-4 mb-3 text-xs text-gray-500">
        <span>Request: <span className="font-mono text-gray-700">{log.requestId}</span></span>
        <span>Model: <span className="font-medium text-gray-700">{log.model}</span></span>
        <span>Key: <span className="font-mono text-gray-700">{log.virtualKey}</span></span>
        <span>Type: <span className="text-gray-700">{log.requestType}</span></span>
      </div>

      {/* Input snippet */}
      <div className="mb-3">
        <div className="text-xs font-medium text-gray-500 mb-1">Input that triggered violation</div>
        <div className="bg-gray-50 rounded p-2 text-xs font-mono text-gray-700 leading-relaxed">
          {log.inputSnippet}
        </div>
      </div>

      {/* Why it failed */}
      <div className="mb-3">
        <div className="text-xs font-medium text-red-600 mb-1">Why this failed</div>
        <div className="text-sm text-gray-700 leading-relaxed">
          {log.violationReason}
        </div>
      </div>

      {/* How to fix */}
      <div>
        <div className="text-xs font-medium text-green-700 mb-1">Recommended fix</div>
        <div className="text-sm text-gray-700 leading-relaxed">
          {log.recommendation}
        </div>
      </div>
    </Card>
  );

  return (
    <div>
      {/* Date Picker */}
      <div className="flex justify-end mb-4">
        <AdvancedDatePicker value={dateValue} onValueChange={setDateValue} />
      </div>

      {/* KPI Cards */}
      <Col numColSpan={2}>
        <Card className="mt-4">
          <Title>Compliance Metrics</Title>
          <Grid numItems={5} className="gap-4 mt-4">
            <Card>
              <Title>Total Requests</Title>
              <Text className="text-2xl font-bold mt-2">
                {MOCK_KPI.totalRequests.toLocaleString()}
              </Text>
            </Card>
            <Card>
              <Title>EU AI Act Violations</Title>
              <Text className="text-2xl font-bold mt-2 text-red-600">
                {MOCK_KPI.euAiActViolations}
              </Text>
            </Card>
            <Card>
              <Title>GDPR Violations</Title>
              <Text className="text-2xl font-bold mt-2 text-red-600">
                {MOCK_KPI.gdprViolations}
              </Text>
            </Card>
            <Card>
              <Title>MCP Unregistered Calls</Title>
              <Text className="text-2xl font-bold mt-2 text-orange-500">
                {MOCK_KPI.mcpUnregisteredCalls}
              </Text>
            </Card>
            <Card>
              <Title>Compliant Requests</Title>
              <Text className="text-2xl font-bold mt-2 text-green-600">
                {MOCK_KPI.compliantRequests.toLocaleString()}
              </Text>
            </Card>
          </Grid>
        </Card>
      </Col>

      {/* Daily Violations Chart */}
      <Col numColSpan={2}>
        <Card className="mt-4">
          <Title>Daily Violations</Title>
          <BarChart
            className="mt-4"
            data={MOCK_DAILY_VIOLATIONS}
            index="date"
            categories={["Compliant", "EU AI Act", "GDPR", "MCP Unregistered"]}
            colors={["cyan", "red", "orange", "amber"]}
            stack={true}
            yAxisWidth={60}
            showLegend={true}
          />
        </Card>
      </Col>

      {/* Two side-by-side: Teams table + Regulation articles chart */}
      <Grid numItems={2} className="gap-4 mt-4">
        <Col numColSpan={1}>
          <Card className="h-full">
            <div className="flex justify-between items-center mb-4">
              <Title>Top Teams by Violations</Title>
              <Segmented
                options={[
                  { label: "5", value: 5 },
                  { label: "10", value: 10 },
                  { label: "25", value: 25 },
                  { label: "50", value: 50 },
                ]}
                value={teamPageSize}
                onChange={(value) => setTeamPageSize(value as number)}
              />
            </div>
            <Table
              dataSource={MOCK_TEAM_VIOLATIONS}
              columns={teamColumns}
              pagination={{ pageSize: teamPageSize }}
              size="small"
            />
          </Card>
        </Col>

        <Col numColSpan={1}>
          <Card className="h-full">
            <Title>Violations by Regulation Article</Title>
            <BarChart
              className="mt-4"
              data={MOCK_REGULATION_ARTICLES}
              index="article"
              categories={["count"]}
              colors={["cyan"]}
              layout="vertical"
              yAxisWidth={220}
              showLegend={false}
            />
          </Card>
        </Col>
      </Grid>

      {/* Violations by Request Type */}
      <Col numColSpan={2}>
        <Card className="mt-4">
          <Title>Violations by Request Type</Title>
          <BarChart
            className="mt-4"
            data={MOCK_REQUEST_TYPE_VIOLATIONS}
            index="type"
            categories={["violations"]}
            colors={["cyan"]}
            layout="vertical"
            yAxisWidth={150}
            showLegend={false}
          />
        </Card>
      </Col>

      {/* Drill-down Drawer */}
      <Drawer
        title={selectedTeam ? `Non-Compliant Requests: ${selectedTeam}` : ""}
        placement="right"
        width={780}
        open={drawerOpen}
        onClose={() => setDrawerOpen(false)}
      >
        {selectedTeam && (
          <div>
            {/* Summary banner */}
            {(() => {
              const logs = MOCK_DRILL_DOWN[selectedTeam] || [];
              const critical = logs.filter(l => l.severity === "critical").length;
              const high = logs.filter(l => l.severity === "high").length;
              const medium = logs.filter(l => l.severity === "medium").length;
              return (
                <div className="flex gap-4 mb-4 p-3 bg-gray-50 rounded-lg text-sm">
                  <span className="text-gray-600">{logs.length} violations total</span>
                  {critical > 0 && <span className="text-red-600 font-medium">{critical} critical</span>}
                  {high > 0 && <span className="text-orange-600 font-medium">{high} high</span>}
                  {medium > 0 && <span className="text-yellow-700 font-medium">{medium} medium</span>}
                </div>
              );
            })()}
            {(MOCK_DRILL_DOWN[selectedTeam] || []).map(renderViolationCard)}
          </div>
        )}
      </Drawer>
    </div>
  );
};

export default PolicyComplianceTab;
