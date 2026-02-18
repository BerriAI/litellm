import React, { useMemo, useState } from "react";
import {
  BarChart,
  Card,
  Col,
  DateRangePickerValue,
  Grid,
  Tab,
  TabGroup,
  Table as TremorTable,
  TableBody,
  TableCell,
  TableHead,
  TableHeaderCell,
  TableRow,
  TabList,
  TabPanel,
  TabPanels,
  Text,
  Title,
} from "@tremor/react";
import { Table, Segmented, Drawer, Select, Breadcrumb } from "antd";
import type { ColumnsType } from "antd/es/table";
import {
  BarChartOutlined,
  GlobalOutlined,
  TeamOutlined,
  KeyOutlined,
  UserOutlined,
} from "@ant-design/icons";
import AdvancedDatePicker from "../../shared/advanced_date_picker";

// ─── Types ──────────────────────────────────────────────────────────────────

type ComplianceView = "global" | "team" | "key" | "user";

interface ViolationLog {
  key: string;
  timestamp: string;
  requestId: string;
  regulation: "EU AI Act" | "GDPR" | "MCP Unregistered";
  article: string;
  severity: "critical" | "high" | "medium";
  model: string;
  requestType: "LLM Call" | "MCP Tool Call" | "Agent Call";
  inputSnippet: string;
  violationReason: string;
  recommendation: string;
  team: string;
  keyAlias: string;
  user: string;
}

interface DailyViolation {
  date: string;
  Compliant: number;
  "EU AI Act": number;
  GDPR: number;
  "MCP Unregistered": number;
}

// ─── Mock Data ──────────────────────────────────────────────────────────────

const USERS = ["ishaan@berri.ai", "sameer@berri.ai", "krrish@berri.ai"];
const TEAMS = [
  "HR Automation Bot",
  "Internal Doc Search",
  "Contract Analyzer",
  "Customer Support",
  "Platform Chatbot",
  "Data Analytics Pipeline",
  "Marketing Content Gen",
  "Code Review Assistant",
];
const KEYS = [
  "sk-hr-bot-prod", "sk-hr-bot-staging",
  "sk-docsearch-prod", "sk-docsearch-dev",
  "sk-contracts-prod",
  "sk-support-prod", "sk-support-eu",
  "sk-chatbot-prod",
  "sk-analytics-prod",
  "sk-marketing-prod", "sk-marketing-staging",
  "sk-codereview-prod",
];

const ALL_VIOLATIONS: ViolationLog[] = [
  // HR Automation Bot — ishaan@berri.ai
  {
    key: "v1", timestamp: "2026-02-17 09:12:34", requestId: "req_8f3a1b2c",
    regulation: "EU AI Act", article: "Art. 5 (Prohibited Practices)", severity: "critical",
    model: "gpt-4o", requestType: "LLM Call", team: "HR Automation Bot", keyAlias: "sk-hr-bot-prod", user: "ishaan@berri.ai",
    inputSnippet: "Based on the employee's performance score of 2.1/10 and attendance record, generate a termination letter and notify HR to proceed with dismissal...",
    violationReason: "Automated decision-making on employment termination without mandatory human oversight. Art. 5(1)(c) prohibits AI systems that evaluate or classify persons based on social behavior leading to detrimental treatment.",
    recommendation: "Add human-in-the-loop approval before any employment decisions. Route output to HR manager for review before action.",
  },
  {
    key: "v2", timestamp: "2026-02-17 08:45:12", requestId: "req_2d4e6f8a",
    regulation: "GDPR", article: "Art. 32 (Data Protection)", severity: "critical",
    model: "gpt-4o", requestType: "LLM Call", team: "HR Automation Bot", keyAlias: "sk-hr-bot-prod", user: "ishaan@berri.ai",
    inputSnippet: "Employee record: Name: John Smith, SSN: 412-55-8901, DOB: 1985-03-14, Medical leave history: 3 instances of mental health leave in 2025...",
    violationReason: "Unencrypted PII (SSN, date of birth) and special category data (health records) sent to external LLM provider without data protection measures.",
    recommendation: "Mask or tokenize PII before sending to LLM. Use litellm guardrails to detect and redact sensitive fields (pii_masking). Never send health data to external providers.",
  },
  {
    key: "v3", timestamp: "2026-02-16 14:22:08", requestId: "req_9c1d3e5f",
    regulation: "GDPR", article: "Art. 32 (Data Protection)", severity: "high",
    model: "claude-3-5-sonnet", requestType: "LLM Call", team: "HR Automation Bot", keyAlias: "sk-hr-bot-prod", user: "ishaan@berri.ai",
    inputSnippet: "Analyser le dossier de Marie Dupont: adresse 12 rue de la Paix Paris, numero secu 2 85 03 75 108 042 15, evaluations de performance 2024-2025...",
    violationReason: "French national ID number (numero de securite sociale) and home address transmitted to LLM without consent or encryption.",
    recommendation: "Enable PII guardrail for French ID patterns. Require explicit consent before processing employee evaluations with AI.",
  },
  {
    key: "v4", timestamp: "2026-02-16 11:03:55", requestId: "req_4b6c8d0e",
    regulation: "EU AI Act", article: "Art. 5 (Prohibited Practices)", severity: "critical",
    model: "gpt-4o", requestType: "LLM Call", team: "HR Automation Bot", keyAlias: "sk-hr-bot-prod", user: "ishaan@berri.ai",
    inputSnippet: "Rank all employees in the engineering department by: productivity score, peer review sentiment, Slack activity metrics, badge-in frequency. Flag bottom 10% for performance improvement plan...",
    violationReason: "Social scoring of employees using behavioral surveillance data (Slack activity, badge-in frequency). This constitutes prohibited social scoring under Art. 5(1)(c).",
    recommendation: "Remove behavioral surveillance inputs. Performance reviews must use only job-relevant, transparent criteria with employee awareness.",
  },
  {
    key: "v5", timestamp: "2026-02-15 16:47:21", requestId: "req_7a9b1c3d",
    regulation: "GDPR", article: "Art. 32 (Data Protection)", severity: "high",
    model: "gpt-4o", requestType: "LLM Call", team: "HR Automation Bot", keyAlias: "sk-hr-bot-prod", user: "ishaan@berri.ai",
    inputSnippet: "Summarize sick leave patterns for the following employees and flag anyone with >5 days mental health leave: [list of 47 employees with full medical records]...",
    violationReason: "Bulk processing of health data (special category under Art. 9 GDPR) without explicit consent or legitimate basis.",
    recommendation: "Health data processing requires explicit employee consent per Art. 9(2)(a). Aggregate and anonymize before any AI analysis. Consider EU-hosted model.",
  },
  // HR Automation Bot — sameer@berri.ai (staging key)
  {
    key: "v6", timestamp: "2026-02-15 10:18:44", requestId: "req_aa1b2c3d",
    regulation: "EU AI Act", article: "Art. 5 (Prohibited Practices)", severity: "high",
    model: "gpt-4o", requestType: "LLM Call", team: "HR Automation Bot", keyAlias: "sk-hr-bot-staging", user: "sameer@berri.ai",
    inputSnippet: "Generate performance summary for candidates based on their social media profiles, LinkedIn endorsements, and inferred personality traits...",
    violationReason: "Using social media data and inferred personality traits for employment decisions constitutes prohibited social scoring.",
    recommendation: "Remove social media analysis from hiring pipeline. Use only job-relevant assessments with candidate consent.",
  },
  {
    key: "v7", timestamp: "2026-02-14 15:33:22", requestId: "req_bb2c3d4e",
    regulation: "GDPR", article: "Art. 32 (Data Protection)", severity: "high",
    model: "claude-3-5-sonnet", requestType: "LLM Call", team: "HR Automation Bot", keyAlias: "sk-hr-bot-staging", user: "sameer@berri.ai",
    inputSnippet: "Process these 12 employee medical certificates for leave validation. Documents include physician names, diagnoses, and recommended treatment plans...",
    violationReason: "Medical certificates with diagnoses and treatment plans (special category data) processed without explicit consent or adequate safeguards.",
    recommendation: "Implement data minimization — only send leave dates and approval status to AI. Keep medical details in secured HR system only.",
  },
  // Internal Doc Search — krrish@berri.ai
  {
    key: "v8", timestamp: "2026-02-17 10:05:18", requestId: "req_1e2f3a4b",
    regulation: "GDPR", article: "Art. 32 (Data Protection)", severity: "high",
    model: "text-embedding-3-small", requestType: "LLM Call", team: "Internal Doc Search", keyAlias: "sk-docsearch-prod", user: "krrish@berri.ai",
    inputSnippet: "Search query: 'Find all contracts mentioning employee salary bands for Sarah Chen, Michael Rodriguez, and compensation packages above 200k'...",
    violationReason: "Search query retrieves and exposes individual salary data (personal data) without access controls or legitimate business need verification.",
    recommendation: "Add role-based access controls to document search. Salary data queries should require manager-level permissions and audit logging.",
  },
  {
    key: "v9", timestamp: "2026-02-16 09:33:41", requestId: "req_5c6d7e8f",
    regulation: "MCP Unregistered", article: "MCP Unregistered Server", severity: "medium",
    model: "gpt-4o", requestType: "MCP Tool Call", team: "Internal Doc Search", keyAlias: "sk-docsearch-prod", user: "krrish@berri.ai",
    inputSnippet: "Tool call to 'internal-search-v2' server at endpoint https://search-staging.internal:8443/query — server not found in MCP registry...",
    violationReason: "MCP tool call routed to unregistered server 'internal-search-v2'. This server is not in the approved MCP registry and has not been security-reviewed.",
    recommendation: "Register 'internal-search-v2' in the MCP server registry via Settings > MCP Servers. Ensure security review is completed before production use.",
  },
  {
    key: "v10", timestamp: "2026-02-15 15:22:09", requestId: "req_9a0b1c2d",
    regulation: "GDPR", article: "Art. 32 (Data Protection)", severity: "high",
    model: "text-embedding-3-small", requestType: "LLM Call", team: "Internal Doc Search", keyAlias: "sk-docsearch-prod", user: "krrish@berri.ai",
    inputSnippet: "Recherche: 'dossiers medicaux employes site Lyon, certificats arret maladie 2025, notes medecin du travail'...",
    violationReason: "Search query targets medical records (special category data). Embedding model processes sensitive health information without adequate protection.",
    recommendation: "Exclude medical/health document collections from general search index. Create separate, access-controlled index with explicit consent requirements.",
  },
  // Internal Doc Search — ishaan@berri.ai (dev key)
  {
    key: "v11", timestamp: "2026-02-16 11:44:21", requestId: "req_dd4e5f6a",
    regulation: "GDPR", article: "Art. 32 (Data Protection)", severity: "medium",
    model: "text-embedding-3-small", requestType: "LLM Call", team: "Internal Doc Search", keyAlias: "sk-docsearch-dev", user: "ishaan@berri.ai",
    inputSnippet: "Index all documents in /shared/hr/personnel-files/ including performance reviews, salary letters, and disciplinary records...",
    violationReason: "Bulk indexing of personnel files containing personal data without consent or data protection impact assessment.",
    recommendation: "Conduct a DPIA before indexing personnel files. Implement access controls and audit logging for sensitive document collections.",
  },
  {
    key: "v12", timestamp: "2026-02-15 09:12:33", requestId: "req_ee5f6a7b",
    regulation: "MCP Unregistered", article: "MCP Unregistered Server", severity: "medium",
    model: "gpt-4o", requestType: "MCP Tool Call", team: "Internal Doc Search", keyAlias: "sk-docsearch-dev", user: "ishaan@berri.ai",
    inputSnippet: "Tool call to 'dev-search-experimental' at localhost:9200/query — server not in MCP registry...",
    violationReason: "Development MCP server 'dev-search-experimental' used in staging environment without being registered in the MCP registry.",
    recommendation: "Register all MCP servers including development instances. Use environment-specific registries for dev/staging/prod.",
  },
  // Contract Analyzer — sameer@berri.ai
  {
    key: "v13", timestamp: "2026-02-17 07:55:02", requestId: "req_3d4e5f6a",
    regulation: "EU AI Act", article: "Art. 9 (Risk Management)", severity: "high",
    model: "claude-3-5-sonnet", requestType: "LLM Call", team: "Contract Analyzer", keyAlias: "sk-contracts-prod", user: "sameer@berri.ai",
    inputSnippet: "Analyze this $4.2M vendor contract and recommend whether to approve or reject. Key terms: liability cap, SLA penalties, data processing addendum. Auto-approve if risk score < 0.3...",
    violationReason: "High-risk AI decision (contract approval >$1M) without mandatory risk assessment documentation. Art. 9 requires documented risk management for high-value automated decisions.",
    recommendation: "Contracts above threshold must go through documented risk assessment. Add human approval step for AI-recommended contract decisions above $1M.",
  },
  {
    key: "v14", timestamp: "2026-02-16 13:18:45", requestId: "req_7b8c9d0e",
    regulation: "GDPR", article: "Art. 38 (Audit Records)", severity: "medium",
    model: "gpt-4o", requestType: "LLM Call", team: "Contract Analyzer", keyAlias: "sk-contracts-prod", user: "sameer@berri.ai",
    inputSnippet: "Extract all personal data subjects mentioned in the attached data processing agreement. List names, roles, and data categories processed...",
    violationReason: "Contract analysis extracting personal data without maintaining required audit records. Art. 38 requires DPO notification and logging for data subject identification activities.",
    recommendation: "Enable detailed audit logging for all contract analysis requests involving personal data. Notify DPO when data subject identification is performed.",
  },
  // Customer Support — krrish@berri.ai
  {
    key: "v15", timestamp: "2026-02-14 11:22:33", requestId: "req_1f2a3b4c",
    regulation: "EU AI Act", article: "Art. 12 (Transparency)", severity: "medium",
    model: "gpt-4o-mini", requestType: "LLM Call", team: "Customer Support", keyAlias: "sk-support-prod", user: "krrish@berri.ai",
    inputSnippet: "Customer asked: 'Am I speaking with a real person?' System prompt instructs: 'You are a helpful customer service representative named Alex. Never reveal you are an AI.'...",
    violationReason: "AI system instructed to conceal its nature when directly asked by user. Art. 12 requires AI systems to be transparent about their non-human nature.",
    recommendation: "Update system prompt to disclose AI nature when asked. Add standard disclosure: 'I'm an AI assistant powered by [company]. I can connect you with a human agent.'",
  },
  {
    key: "v16", timestamp: "2026-02-13 09:44:17", requestId: "req_5d6e7f8a",
    regulation: "EU AI Act", article: "Art. 12 (Transparency)", severity: "medium",
    model: "gpt-4o-mini", requestType: "LLM Call", team: "Customer Support", keyAlias: "sk-support-prod", user: "krrish@berri.ai",
    inputSnippet: "Le client demande: 'Est-ce que je parle a un humain ou a un robot?' Instruction systeme: 'Repondre comme un agent humain, ne pas mentionner l'IA'...",
    violationReason: "Same transparency violation in French-language support channel. Customer explicitly asked if speaking to AI and system is instructed to deny it.",
    recommendation: "Apply the same transparency fix across all language channels. System prompt must allow AI self-identification in all supported languages.",
  },
  // Customer Support — sameer@berri.ai (EU key)
  {
    key: "v17", timestamp: "2026-02-15 14:05:19", requestId: "req_ff6a7b8c",
    regulation: "GDPR", article: "Art. 32 (Data Protection)", severity: "medium",
    model: "gpt-4o-mini", requestType: "LLM Call", team: "Customer Support", keyAlias: "sk-support-eu", user: "sameer@berri.ai",
    inputSnippet: "Customer ticket #4821: 'My account email is hans.weber@gmail.com, phone +49 151 12345678. I need to update my billing address to Hauptstraße 42, 80331 München'...",
    violationReason: "Customer PII (email, phone, address) included verbatim in LLM prompt without masking.",
    recommendation: "Enable PII masking guardrail for support channel. Mask email, phone, and address before sending to LLM.",
  },
  // Platform Chatbot — ishaan@berri.ai
  {
    key: "v18", timestamp: "2026-02-12 16:08:52", requestId: "req_9b0c1d2e",
    regulation: "EU AI Act", article: "Art. 12 (Transparency)", severity: "medium",
    model: "gpt-4o-mini", requestType: "Agent Call", team: "Platform Chatbot", keyAlias: "sk-chatbot-prod", user: "ishaan@berri.ai",
    inputSnippet: "Chatbot greeting: 'Hi! I'm your personal assistant. How can I help you today?' — no AI disclosure in greeting or system prompt...",
    violationReason: "Public-facing chatbot does not identify itself as an AI system at any point in the interaction. Art. 12 requires clear disclosure before or at the start of interaction.",
    recommendation: "Add AI disclosure to chatbot greeting: 'Hi! I'm an AI assistant for [Platform]. How can I help?' Also add disclosure in the chat widget UI.",
  },
];

const MOCK_DAILY_VIOLATIONS: DailyViolation[] = [
  { date: "Feb 10", Compliant: 548, "EU AI Act": 2, GDPR: 5, "MCP Unregistered": 1 },
  { date: "Feb 11", Compliant: 612, "EU AI Act": 3, GDPR: 6, "MCP Unregistered": 1 },
  { date: "Feb 12", Compliant: 655, "EU AI Act": 4, GDPR: 7, "MCP Unregistered": 2 },
  { date: "Feb 13", Compliant: 670, "EU AI Act": 3, GDPR: 4, "MCP Unregistered": 0 },
  { date: "Feb 14", Compliant: 660, "EU AI Act": 3, GDPR: 5, "MCP Unregistered": 1 },
  { date: "Feb 15", Compliant: 675, "EU AI Act": 2, GDPR: 6, "MCP Unregistered": 2 },
  { date: "Feb 16", Compliant: 540, "EU AI Act": 4, GDPR: 6, "MCP Unregistered": 1 },
  { date: "Feb 17", Compliant: 391, "EU AI Act": 2, GDPR: 2, "MCP Unregistered": 0 },
];

const MOCK_TEAM_REQUESTS: Record<string, number> = {
  "HR Automation Bot": 412,
  "Internal Doc Search": 1089,
  "Contract Analyzer": 287,
  "Customer Support": 1832,
  "Platform Chatbot": 523,
  "Data Analytics Pipeline": 345,
  "Marketing Content Gen": 198,
  "Code Review Assistant": 137,
};

const MOCK_KEY_OWNERS: Record<string, { team: string; user: string }> = {
  "sk-hr-bot-prod": { team: "HR Automation Bot", user: "ishaan@berri.ai" },
  "sk-hr-bot-staging": { team: "HR Automation Bot", user: "sameer@berri.ai" },
  "sk-docsearch-prod": { team: "Internal Doc Search", user: "krrish@berri.ai" },
  "sk-docsearch-dev": { team: "Internal Doc Search", user: "ishaan@berri.ai" },
  "sk-contracts-prod": { team: "Contract Analyzer", user: "sameer@berri.ai" },
  "sk-support-prod": { team: "Customer Support", user: "krrish@berri.ai" },
  "sk-support-eu": { team: "Customer Support", user: "sameer@berri.ai" },
  "sk-chatbot-prod": { team: "Platform Chatbot", user: "ishaan@berri.ai" },
  "sk-analytics-prod": { team: "Data Analytics Pipeline", user: "ishaan@berri.ai" },
  "sk-marketing-prod": { team: "Marketing Content Gen", user: "krrish@berri.ai" },
  "sk-marketing-staging": { team: "Marketing Content Gen", user: "sameer@berri.ai" },
  "sk-codereview-prod": { team: "Code Review Assistant", user: "krrish@berri.ai" },
};

// ─── Helpers ────────────────────────────────────────────────────────────────

function groupBy<T>(arr: T[], fn: (item: T) => string): Record<string, T[]> {
  const result: Record<string, T[]> = {};
  for (const item of arr) {
    const k = fn(item);
    if (!result[k]) result[k] = [];
    result[k].push(item);
  }
  return result;
}

function countByRegulation(violations: ViolationLog[]) {
  let euAiAct = 0, gdpr = 0, mcp = 0;
  for (const v of violations) {
    if (v.regulation === "EU AI Act") euAiAct++;
    else if (v.regulation === "GDPR") gdpr++;
    else mcp++;
  }
  return { euAiAct, gdpr, mcp, total: euAiAct + gdpr + mcp };
}

function riskLevel(total: number): "HIGH" | "MED" | "LOW" | "NONE" {
  if (total === 0) return "NONE";
  if (total >= 5) return "HIGH";
  if (total >= 2) return "MED";
  return "LOW";
}

// ─── Shared UI pieces ───────────────────────────────────────────────────────

const violationCount = (val: number) => {
  if (val === 0) return <span className="text-gray-300 font-normal">0</span>;
  if (val > 5) return <span className="text-red-600 font-semibold">{val}</span>;
  return <span className="text-orange-500 font-medium">{val}</span>;
};

const riskBadge = (risk: string) => {
  const styles: Record<string, string> = {
    HIGH: "bg-red-50 text-red-600 border border-red-200",
    MED: "bg-orange-50 text-orange-600 border border-orange-200",
    LOW: "bg-yellow-50 text-yellow-700 border border-yellow-200",
    NONE: "bg-green-50 text-green-600 border border-green-200",
  };
  return (
    <span className={`inline-block px-2 py-0.5 rounded-full text-xs font-medium ${styles[risk] || "bg-gray-100 text-gray-500"}`}>
      {risk === "NONE" ? "COMPLIANT" : risk}
    </span>
  );
};

const severityBadge = (severity: string) => {
  const styles: Record<string, string> = {
    critical: "bg-red-50 text-red-600 border border-red-200",
    high: "bg-orange-50 text-orange-600 border border-orange-200",
    medium: "bg-yellow-50 text-yellow-700 border border-yellow-200",
  };
  return (
    <span className={`inline-block px-2 py-0.5 rounded-full text-xs font-medium ${styles[severity] || ""}`}>
      {severity.toUpperCase()}
    </span>
  );
};

const regulationBadge = (reg: string) => {
  const styles: Record<string, string> = {
    "EU AI Act": "bg-indigo-50 text-indigo-700 border border-indigo-200",
    GDPR: "bg-green-50 text-green-700 border border-green-200",
    "MCP Unregistered": "bg-orange-50 text-orange-700 border border-orange-200",
  };
  return (
    <span className={`inline-block px-2 py-0.5 rounded-full text-xs font-medium ${styles[reg] || ""}`}>
      {reg}
    </span>
  );
};

const ViolationCard = ({ log }: { log: ViolationLog }) => (
  <Card key={log.key} className="mb-3">
    <div className="flex items-center justify-between mb-3">
      <div className="flex items-center gap-2 flex-wrap">
        {severityBadge(log.severity)}
        {regulationBadge(log.regulation)}
        <span className="text-xs text-gray-400">{log.article}</span>
      </div>
      <span className="text-xs text-gray-400">{log.timestamp}</span>
    </div>

    <div className="flex gap-4 mb-3 text-xs text-gray-500 flex-wrap">
      <span>Request: <span className="font-mono text-gray-700">{log.requestId}</span></span>
      <span>Model: <span className="font-medium text-gray-700">{log.model}</span></span>
      <span>Key: <span className="font-mono text-gray-700">{log.keyAlias}</span></span>
      <span>User: <span className="font-medium text-gray-700">{log.user}</span></span>
      <span>Type: <span className="text-gray-700">{log.requestType}</span></span>
    </div>

    <div className="mb-3">
      <div className="text-xs font-medium text-gray-500 mb-1">Input that triggered violation</div>
      <div className="bg-gray-50 rounded p-2 text-xs font-mono text-gray-700 leading-relaxed">{log.inputSnippet}</div>
    </div>

    <div className="mb-3">
      <div className="text-xs font-medium text-red-600 mb-1">Why this failed</div>
      <div className="text-sm text-gray-700 leading-relaxed">{log.violationReason}</div>
    </div>

    <div>
      <div className="text-xs font-medium text-green-700 mb-1">Recommended fix</div>
      <div className="text-sm text-gray-700 leading-relaxed">{log.recommendation}</div>
    </div>
  </Card>
);

// ─── View Selector (mirrors UsageViewSelect) ───────────────────────────────

const VIEW_OPTIONS: { value: ComplianceView; label: string; description: string; icon: React.ReactNode }[] = [
  { value: "global", label: "Global Compliance", description: "View compliance across all resources", icon: <GlobalOutlined style={{ fontSize: "16px" }} /> },
  { value: "team", label: "Team Compliance", description: "View compliance by team", icon: <TeamOutlined style={{ fontSize: "16px" }} /> },
  { value: "key", label: "Key Compliance", description: "View compliance by virtual key", icon: <KeyOutlined style={{ fontSize: "16px" }} /> },
  { value: "user", label: "User Compliance", description: "View compliance by user", icon: <UserOutlined style={{ fontSize: "16px" }} /> },
];

const ComplianceViewSelect = ({
  value,
  onChange,
}: {
  value: ComplianceView;
  onChange: (v: ComplianceView) => void;
}) => {
  return (
    <div className="w-full">
      <div className="flex flex-wrap items-center justify-start gap-4">
        <div className="flex items-stretch gap-2 min-w-0">
          <div className="flex-shrink-0 flex items-center">
            <BarChartOutlined style={{ fontSize: "32px" }} />
          </div>
          <div className="flex-1 min-w-0">
            <h3 className="text-sm font-semibold text-gray-900 mb-0.5 leading-tight">Compliance View</h3>
            <p className="text-xs text-gray-600 leading-tight">Select the compliance data you want to view</p>
          </div>
        </div>
        <div className="flex-shrink-0">
          <Select
            value={value}
            onChange={onChange}
            className="w-54 sm:w-64 md:w-72"
            size="large"
            options={VIEW_OPTIONS.map((opt) => ({ value: opt.value, label: opt.label }))}
            optionRender={(option) => {
              const opt = VIEW_OPTIONS.find((o) => o.value === option.value);
              if (!opt) return option.label;
              return (
                <div className="flex items-center gap-2 py-1">
                  <div className="flex-shrink-0 mt-0.5">{opt.icon}</div>
                  <div className="flex-1 min-w-0">
                    <div className="text-sm font-medium text-gray-900">{opt.label}</div>
                    <div className="text-xs text-gray-600 mt-0.5">{opt.description}</div>
                  </div>
                </div>
              );
            }}
            labelRender={(props) => {
              const opt = VIEW_OPTIONS.find((o) => o.value === props.value);
              if (!opt) return props.label;
              return (
                <div className="flex items-center gap-2">
                  <div>{opt.icon}</div>
                  <span className="text-sm">{opt.label}</span>
                </div>
              );
            }}
          />
        </div>
      </div>
    </div>
  );
};

// ─── KPI Row (shared across views) ─────────────────────────────────────────

const KPIRow = ({ violations, totalRequests }: { violations: ViolationLog[]; totalRequests: number }) => {
  const counts = countByRegulation(violations);
  const compliant = totalRequests - counts.total;
  return (
    <Card>
      <Title>Compliance Metrics</Title>
      <Grid numItems={5} className="gap-4 mt-4">
        <Card><Title>Total Requests</Title><Text className="text-2xl font-bold mt-2">{totalRequests.toLocaleString()}</Text></Card>
        <Card><Title>EU AI Act Violations</Title><Text className="text-2xl font-bold mt-2 text-red-600">{counts.euAiAct}</Text></Card>
        <Card><Title>GDPR Violations</Title><Text className="text-2xl font-bold mt-2 text-red-600">{counts.gdpr}</Text></Card>
        <Card><Title>MCP Unregistered</Title><Text className="text-2xl font-bold mt-2 text-orange-500">{counts.mcp}</Text></Card>
        <Card><Title>Compliant Requests</Title><Text className="text-2xl font-bold mt-2 text-green-600">{Math.max(0, compliant).toLocaleString()}</Text></Card>
      </Grid>
    </Card>
  );
};

// ─── Entity breakdown table + chart (mirrors EntityUsage pattern) ───────────

interface EntityRow {
  key: string;
  name: string;
  totalRequests: number;
  compliant: number;
  euAiAct: number;
  gdpr: number;
  mcp: number;
  totalViolations: number;
  risk: string;
}

const EntityBreakdown = ({
  title,
  subtitle,
  rows,
  onRowClick,
  nameColumn,
  limit,
  setLimit,
}: {
  title: string;
  subtitle: string;
  rows: EntityRow[];
  onRowClick?: (row: EntityRow) => void;
  nameColumn: string;
  limit: number;
  setLimit: (n: number) => void;
}) => {
  const chartData = rows.slice(0, 5).map((r) => ({
    name: r.name.length > 20 ? `${r.name.slice(0, 20)}...` : r.name,
    "EU AI Act": r.euAiAct,
    GDPR: r.gdpr,
    "MCP Unregistered": r.mcp,
  }));

  return (
    <Card>
      <div className="flex justify-between items-start mb-4">
        <div>
          <Title>{title}</Title>
          <Text className="text-xs text-gray-500">{subtitle}</Text>
        </div>
        <Segmented
          options={[
            { label: "5", value: 5 },
            { label: "10", value: 10 },
            { label: "All", value: 100 },
          ]}
          value={limit}
          onChange={(v) => setLimit(v as number)}
        />
      </div>
      <Grid numItems={2} className="gap-6">
        <Col numColSpan={1}>
          <BarChart
            className="mt-4 h-52"
            data={chartData}
            index="name"
            categories={["EU AI Act", "GDPR", "MCP Unregistered"]}
            colors={["indigo", "emerald", "amber"]}
            layout="vertical"
            yAxisWidth={160}
            stack={true}
            showLegend={true}
          />
        </Col>
        <Col numColSpan={1}>
          <div className="max-h-[300px] overflow-y-auto">
            <TremorTable>
              <TableHead>
                <TableRow>
                  <TableHeaderCell>{nameColumn}</TableHeaderCell>
                  <TableHeaderCell>Requests</TableHeaderCell>
                  <TableHeaderCell className="text-green-600">Compliant</TableHeaderCell>
                  <TableHeaderCell className="text-red-600">Violations</TableHeaderCell>
                  <TableHeaderCell>Risk</TableHeaderCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {rows.slice(0, limit).map((row) => (
                  <TableRow key={row.key}>
                    <TableCell>
                      {onRowClick ? (
                        <a onClick={() => onRowClick(row)} className="text-blue-600 hover:text-blue-800 cursor-pointer font-medium">
                          {row.name}
                        </a>
                      ) : (
                        row.name
                      )}
                    </TableCell>
                    <TableCell>{row.totalRequests.toLocaleString()}</TableCell>
                    <TableCell className="text-green-600">{row.compliant.toLocaleString()}</TableCell>
                    <TableCell className="text-red-600">{row.totalViolations > 0 ? row.totalViolations : <span className="text-gray-300">0</span>}</TableCell>
                    <TableCell>{riskBadge(row.risk)}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </TremorTable>
          </div>
        </Col>
      </Grid>
    </Card>
  );
};

// ─── Regulation article breakdown ───────────────────────────────────────────

const RegulationBreakdown = ({ violations }: { violations: ViolationLog[] }) => {
  const byArticle = groupBy(violations, (v) => v.article);
  const data = Object.entries(byArticle)
    .map(([article, vs]) => ({ article, count: vs.length }))
    .sort((a, b) => b.count - a.count);

  return (
    <Card>
      <Title>Violations by Regulation Article</Title>
      <BarChart
        className="mt-4"
        data={data}
        index="article"
        categories={["count"]}
        colors={["cyan"]}
        layout="vertical"
        yAxisWidth={220}
        showLegend={false}
      />
    </Card>
  );
};

const RequestTypeBreakdown = ({ violations }: { violations: ViolationLog[] }) => {
  const byType = groupBy(violations, (v) => v.requestType);
  const data = Object.entries(byType)
    .map(([type, vs]) => ({ type, violations: vs.length }))
    .sort((a, b) => b.violations - a.violations);

  return (
    <Card>
      <Title>Violations by Request Type</Title>
      <BarChart
        className="mt-4"
        data={data}
        index="type"
        categories={["violations"]}
        colors={["cyan"]}
        layout="vertical"
        yAxisWidth={150}
        showLegend={false}
      />
    </Card>
  );
};

// ─── Violation detail drawer content ────────────────────────────────────────

const ViolationListPanel = ({
  violations,
  emptyMessage,
}: {
  violations: ViolationLog[];
  emptyMessage: string;
}) => {
  const critical = violations.filter((v) => v.severity === "critical").length;
  const high = violations.filter((v) => v.severity === "high").length;
  const medium = violations.filter((v) => v.severity === "medium").length;

  if (violations.length === 0) {
    return (
      <div className="text-center py-8 text-gray-400">
        <div className="text-lg mb-1">No violations</div>
        <div className="text-sm">{emptyMessage}</div>
      </div>
    );
  }

  return (
    <div>
      <div className="flex gap-4 mb-4 p-3 bg-gray-50 rounded-lg text-sm flex-wrap">
        <span className="text-gray-600"><span className="font-medium">{violations.length}</span> violations total</span>
        {critical > 0 && <span className="text-red-600 font-medium">{critical} critical</span>}
        {high > 0 && <span className="text-orange-600 font-medium">{high} high</span>}
        {medium > 0 && <span className="text-yellow-700 font-medium">{medium} medium</span>}
      </div>
      {violations.map((v) => <ViolationCard key={v.key} log={v} />)}
    </div>
  );
};

// ─── Global Compliance View ─────────────────────────────────────────────────

const GlobalComplianceView = () => {
  const [entityLimit, setEntityLimit] = useState(10);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [selectedEntity, setSelectedEntity] = useState<{ type: string; name: string } | null>(null);

  const totalRequests = Object.values(MOCK_TEAM_REQUESTS).reduce((a, b) => a + b, 0);
  const byTeam = groupBy(ALL_VIOLATIONS, (v) => v.team);

  const teamRows: EntityRow[] = TEAMS.map((team) => {
    const vs = byTeam[team] || [];
    const counts = countByRegulation(vs);
    const reqs = MOCK_TEAM_REQUESTS[team] || 0;
    return {
      key: team, name: team, totalRequests: reqs, compliant: reqs - counts.total,
      euAiAct: counts.euAiAct, gdpr: counts.gdpr, mcp: counts.mcp,
      totalViolations: counts.total, risk: riskLevel(counts.total),
    };
  }).sort((a, b) => b.totalViolations - a.totalViolations);

  const byKey = groupBy(ALL_VIOLATIONS, (v) => v.keyAlias);
  const keyRows: EntityRow[] = KEYS.map((k) => {
    const vs = byKey[k] || [];
    const counts = countByRegulation(vs);
    return {
      key: k, name: k, totalRequests: 0, compliant: 0,
      euAiAct: counts.euAiAct, gdpr: counts.gdpr, mcp: counts.mcp,
      totalViolations: counts.total, risk: riskLevel(counts.total),
    };
  }).sort((a, b) => b.totalViolations - a.totalViolations);

  const byUser = groupBy(ALL_VIOLATIONS, (v) => v.user);
  const userRows: EntityRow[] = USERS.map((u) => {
    const vs = byUser[u] || [];
    const counts = countByRegulation(vs);
    return {
      key: u, name: u, totalRequests: 0, compliant: 0,
      euAiAct: counts.euAiAct, gdpr: counts.gdpr, mcp: counts.mcp,
      totalViolations: counts.total, risk: riskLevel(counts.total),
    };
  }).sort((a, b) => b.totalViolations - a.totalViolations);

  const drawerViolations = useMemo(() => {
    if (!selectedEntity) return [];
    if (selectedEntity.type === "team") return ALL_VIOLATIONS.filter((v) => v.team === selectedEntity.name);
    if (selectedEntity.type === "key") return ALL_VIOLATIONS.filter((v) => v.keyAlias === selectedEntity.name);
    if (selectedEntity.type === "user") return ALL_VIOLATIONS.filter((v) => v.user === selectedEntity.name);
    return [];
  }, [selectedEntity]);

  return (
    <>
      <TabGroup>
        <TabList variant="solid" className="mt-1">
          <Tab>Violations</Tab>
          <Tab>Team Activity</Tab>
          <Tab>Key Activity</Tab>
          <Tab>User Activity</Tab>
        </TabList>
        <TabPanels>
          {/* Violations tab */}
          <TabPanel>
            <Grid numItems={2} className="gap-2 w-full">
              <Col numColSpan={2}>
                <KPIRow violations={ALL_VIOLATIONS} totalRequests={totalRequests} />
              </Col>
              <Col numColSpan={2}>
                <Card>
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
              <Col numColSpan={2}>
                <EntityBreakdown
                  title="All Teams — Compliance Status"
                  subtitle="Click a team to see its violations"
                  rows={teamRows}
                  onRowClick={(row) => { setSelectedEntity({ type: "team", name: row.name }); setDrawerOpen(true); }}
                  nameColumn="Team"
                  limit={entityLimit}
                  setLimit={setEntityLimit}
                />
              </Col>
              <Col numColSpan={1}>
                <RegulationBreakdown violations={ALL_VIOLATIONS} />
              </Col>
              <Col numColSpan={1}>
                <RequestTypeBreakdown violations={ALL_VIOLATIONS} />
              </Col>
            </Grid>
          </TabPanel>

          {/* Team Activity tab — same entity breakdown as usage page */}
          <TabPanel>
            <Grid numItems={2} className="gap-2 w-full">
              <Col numColSpan={2}>
                <EntityBreakdown
                  title="Violations Per Team"
                  subtitle="All teams shown — click to drill down"
                  rows={teamRows}
                  onRowClick={(row) => { setSelectedEntity({ type: "team", name: row.name }); setDrawerOpen(true); }}
                  nameColumn="Team"
                  limit={entityLimit}
                  setLimit={setEntityLimit}
                />
              </Col>
            </Grid>
          </TabPanel>

          {/* Key Activity tab */}
          <TabPanel>
            <Grid numItems={2} className="gap-2 w-full">
              <Col numColSpan={2}>
                <EntityBreakdown
                  title="Violations Per Virtual Key"
                  subtitle="All keys shown — click to drill down"
                  rows={keyRows}
                  onRowClick={(row) => { setSelectedEntity({ type: "key", name: row.name }); setDrawerOpen(true); }}
                  nameColumn="Virtual Key"
                  limit={entityLimit}
                  setLimit={setEntityLimit}
                />
              </Col>
            </Grid>
          </TabPanel>

          {/* User Activity tab */}
          <TabPanel>
            <Grid numItems={2} className="gap-2 w-full">
              <Col numColSpan={2}>
                <EntityBreakdown
                  title="Violations Per User"
                  subtitle="All users shown — click to drill down"
                  rows={userRows}
                  onRowClick={(row) => { setSelectedEntity({ type: "user", name: row.name }); setDrawerOpen(true); }}
                  nameColumn="User"
                  limit={entityLimit}
                  setLimit={setEntityLimit}
                />
              </Col>
            </Grid>
          </TabPanel>
        </TabPanels>
      </TabGroup>

      <Drawer
        title={selectedEntity ? `Violations: ${selectedEntity.name}` : ""}
        placement="right"
        width={860}
        open={drawerOpen}
        onClose={() => { setDrawerOpen(false); setSelectedEntity(null); }}
      >
        <ViolationListPanel violations={drawerViolations} emptyMessage="Fully compliant — no violations in this period" />
      </Drawer>
    </>
  );
};

// ─── Team Compliance View (mirrors EntityUsage for team) ────────────────────

const TeamComplianceView = () => {
  const [selectedTeam, setSelectedTeam] = useState<string | null>(null);
  const [teamLimit, setTeamLimit] = useState(10);
  const [keyLimit, setKeyLimit] = useState(10);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [drawerKey, setDrawerKey] = useState<string | null>(null);

  const byTeam = groupBy(ALL_VIOLATIONS, (v) => v.team);
  const teamViolations = selectedTeam ? (byTeam[selectedTeam] || []) : ALL_VIOLATIONS;
  const teamReqs = selectedTeam ? (MOCK_TEAM_REQUESTS[selectedTeam] || 0) : Object.values(MOCK_TEAM_REQUESTS).reduce((a, b) => a + b, 0);

  const teamRows: EntityRow[] = TEAMS.map((team) => {
    const vs = byTeam[team] || [];
    const counts = countByRegulation(vs);
    const reqs = MOCK_TEAM_REQUESTS[team] || 0;
    return {
      key: team, name: team, totalRequests: reqs, compliant: reqs - counts.total,
      euAiAct: counts.euAiAct, gdpr: counts.gdpr, mcp: counts.mcp,
      totalViolations: counts.total, risk: riskLevel(counts.total),
    };
  }).sort((a, b) => b.totalViolations - a.totalViolations);

  const byKey = groupBy(teamViolations, (v) => v.keyAlias);
  const keyRows: EntityRow[] = Object.entries(MOCK_KEY_OWNERS)
    .filter(([, info]) => !selectedTeam || info.team === selectedTeam)
    .map(([k, info]) => {
      const vs = byKey[k] || [];
      const counts = countByRegulation(vs);
      return {
        key: k, name: `${k}  (${info.user})`, totalRequests: 0, compliant: 0,
        euAiAct: counts.euAiAct, gdpr: counts.gdpr, mcp: counts.mcp,
        totalViolations: counts.total, risk: riskLevel(counts.total),
      };
    }).sort((a, b) => b.totalViolations - a.totalViolations);

  const drawerViolations = drawerKey
    ? teamViolations.filter((v) => v.keyAlias === drawerKey)
    : [];

  return (
    <>
      <div className="mb-4">
        <Select
          showSearch
          allowClear
          style={{ width: 360 }}
          placeholder="All Teams"
          value={selectedTeam}
          onChange={(v) => setSelectedTeam(v ?? null)}
          options={TEAMS.map((t) => ({ value: t, label: t }))}
        />
      </div>

      <TabGroup>
        <TabList variant="solid" className="mt-1">
          <Tab>Violations</Tab>
          <Tab>Key Activity</Tab>
        </TabList>
        <TabPanels>
          <TabPanel>
            <Grid numItems={2} className="gap-2 w-full">
              <Col numColSpan={2}>
                <KPIRow violations={teamViolations} totalRequests={teamReqs} />
              </Col>
              {/* Violations by Team — always visible at the top */}
              <Col numColSpan={2}>
                <EntityBreakdown
                  title="Violations by Team"
                  subtitle="Click a team to filter — or use the dropdown above"
                  rows={teamRows}
                  onRowClick={(row) => setSelectedTeam(row.name)}
                  nameColumn="Team"
                  limit={teamLimit}
                  setLimit={setTeamLimit}
                />
              </Col>
              <Col numColSpan={2}>
                <EntityBreakdown
                  title={selectedTeam ? `Key Breakdown — ${selectedTeam}` : "Key Breakdown — All Teams"}
                  subtitle="Click a key to see individual violations and fix suggestions"
                  rows={keyRows}
                  onRowClick={(row) => {
                    const rawKey = row.key;
                    setDrawerKey(rawKey);
                    setDrawerOpen(true);
                  }}
                  nameColumn="Virtual Key (Owner)"
                  limit={keyLimit}
                  setLimit={setKeyLimit}
                />
              </Col>
              <Col numColSpan={1}>
                <RegulationBreakdown violations={teamViolations} />
              </Col>
              <Col numColSpan={1}>
                <RequestTypeBreakdown violations={teamViolations} />
              </Col>
            </Grid>
          </TabPanel>
          <TabPanel>
            <Grid numItems={2} className="gap-2 w-full">
              <Col numColSpan={2}>
                <EntityBreakdown
                  title="All Keys — Violation Counts"
                  subtitle="Click a key to see individual violations"
                  rows={keyRows}
                  onRowClick={(row) => { setDrawerKey(row.key); setDrawerOpen(true); }}
                  nameColumn="Virtual Key (Owner)"
                  limit={keyLimit}
                  setLimit={setKeyLimit}
                />
              </Col>
            </Grid>
          </TabPanel>
        </TabPanels>
      </TabGroup>

      <Drawer
        title={drawerKey ? `Violations: ${drawerKey}` : ""}
        placement="right"
        width={860}
        open={drawerOpen}
        onClose={() => { setDrawerOpen(false); setDrawerKey(null); }}
      >
        {drawerKey && (
          <>
            <div className="mb-4 p-3 bg-gray-50 rounded-lg text-sm">
              <span className="text-gray-500">Owner: </span>
              <span className="font-medium">{MOCK_KEY_OWNERS[drawerKey]?.user}</span>
              <span className="mx-2 text-gray-300">|</span>
              <span className="text-gray-500">Team: </span>
              <span className="font-medium">{MOCK_KEY_OWNERS[drawerKey]?.team}</span>
            </div>
            <ViolationListPanel violations={drawerViolations} emptyMessage="This key is fully compliant" />
          </>
        )}
      </Drawer>
    </>
  );
};

// ─── Key Compliance View (mirrors EntityUsage for key) ──────────────────────

const KeyComplianceView = () => {
  const [selectedKey, setSelectedKey] = useState<string | null>(null);
  const [userLimit, setUserLimit] = useState(10);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [drawerUser, setDrawerUser] = useState<string | null>(null);

  const byKey = groupBy(ALL_VIOLATIONS, (v) => v.keyAlias);
  const keyViolations = selectedKey ? (byKey[selectedKey] || []) : ALL_VIOLATIONS;

  const byUser = groupBy(keyViolations, (v) => v.user);
  const userRows: EntityRow[] = USERS.map((u) => {
    const vs = byUser[u] || [];
    const counts = countByRegulation(vs);
    return {
      key: u, name: u, totalRequests: 0, compliant: 0,
      euAiAct: counts.euAiAct, gdpr: counts.gdpr, mcp: counts.mcp,
      totalViolations: counts.total, risk: riskLevel(counts.total),
    };
  }).filter((r) => !selectedKey || r.totalViolations > 0 || MOCK_KEY_OWNERS[selectedKey]?.user === r.name)
    .sort((a, b) => b.totalViolations - a.totalViolations);

  const drawerViolations = drawerUser
    ? keyViolations.filter((v) => v.user === drawerUser)
    : [];

  return (
    <>
      <div className="mb-4">
        <Select
          showSearch
          allowClear
          style={{ width: 360 }}
          placeholder="All Keys"
          value={selectedKey}
          onChange={(v) => setSelectedKey(v ?? null)}
          options={KEYS.map((k) => ({ value: k, label: `${k}  (${MOCK_KEY_OWNERS[k]?.team})` }))}
        />
      </div>

      <TabGroup>
        <TabList variant="solid" className="mt-1">
          <Tab>Violations</Tab>
          <Tab>User Activity</Tab>
        </TabList>
        <TabPanels>
          <TabPanel>
            <Grid numItems={2} className="gap-2 w-full">
              <Col numColSpan={2}>
                <KPIRow violations={keyViolations} totalRequests={0} />
              </Col>
              <Col numColSpan={2}>
                <EntityBreakdown
                  title="User Breakdown"
                  subtitle="Click a user to see their violations and fix suggestions"
                  rows={userRows}
                  onRowClick={(row) => { setDrawerUser(row.name); setDrawerOpen(true); }}
                  nameColumn="User"
                  limit={userLimit}
                  setLimit={setUserLimit}
                />
              </Col>
              <Col numColSpan={1}>
                <RegulationBreakdown violations={keyViolations} />
              </Col>
              <Col numColSpan={1}>
                <RequestTypeBreakdown violations={keyViolations} />
              </Col>
            </Grid>
          </TabPanel>
          <TabPanel>
            <Grid numItems={2} className="gap-2 w-full">
              <Col numColSpan={2}>
                <EntityBreakdown
                  title="All Users — Violation Counts"
                  subtitle="Click a user to see individual violations"
                  rows={userRows}
                  onRowClick={(row) => { setDrawerUser(row.name); setDrawerOpen(true); }}
                  nameColumn="User"
                  limit={userLimit}
                  setLimit={setUserLimit}
                />
              </Col>
            </Grid>
          </TabPanel>
        </TabPanels>
      </TabGroup>

      <Drawer
        title={drawerUser ? `Violations: ${drawerUser}` : ""}
        placement="right"
        width={860}
        open={drawerOpen}
        onClose={() => { setDrawerOpen(false); setDrawerUser(null); }}
      >
        <ViolationListPanel violations={drawerViolations} emptyMessage="This user is fully compliant" />
      </Drawer>
    </>
  );
};

// ─── User Compliance View ───────────────────────────────────────────────────

const UserComplianceView = () => {
  const [selectedUser, setSelectedUser] = useState<string | null>(null);
  const [keyLimit, setKeyLimit] = useState(10);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [drawerKey, setDrawerKey] = useState<string | null>(null);

  const byUser = groupBy(ALL_VIOLATIONS, (v) => v.user);
  const userViolations = selectedUser ? (byUser[selectedUser] || []) : ALL_VIOLATIONS;

  const byKey = groupBy(userViolations, (v) => v.keyAlias);
  const keyRows: EntityRow[] = KEYS
    .filter((k) => !selectedUser || MOCK_KEY_OWNERS[k]?.user === selectedUser || (byKey[k]?.length ?? 0) > 0)
    .map((k) => {
      const vs = byKey[k] || [];
      const counts = countByRegulation(vs);
      return {
        key: k, name: `${k}  (${MOCK_KEY_OWNERS[k]?.team})`, totalRequests: 0, compliant: 0,
        euAiAct: counts.euAiAct, gdpr: counts.gdpr, mcp: counts.mcp,
        totalViolations: counts.total, risk: riskLevel(counts.total),
      };
    }).sort((a, b) => b.totalViolations - a.totalViolations);

  const drawerViolations = drawerKey
    ? userViolations.filter((v) => v.keyAlias === drawerKey)
    : [];

  return (
    <>
      <div className="mb-4">
        <Select
          showSearch
          allowClear
          style={{ width: 360 }}
          placeholder="All Users"
          value={selectedUser}
          onChange={(v) => setSelectedUser(v ?? null)}
          options={USERS.map((u) => ({ value: u, label: u }))}
        />
      </div>

      <TabGroup>
        <TabList variant="solid" className="mt-1">
          <Tab>Violations</Tab>
          <Tab>Key Activity</Tab>
        </TabList>
        <TabPanels>
          <TabPanel>
            <Grid numItems={2} className="gap-2 w-full">
              <Col numColSpan={2}>
                <KPIRow violations={userViolations} totalRequests={0} />
              </Col>
              <Col numColSpan={2}>
                <EntityBreakdown
                  title="Key Breakdown"
                  subtitle="Click a key to see violations and fix suggestions"
                  rows={keyRows}
                  onRowClick={(row) => { setDrawerKey(row.key); setDrawerOpen(true); }}
                  nameColumn="Virtual Key (Team)"
                  limit={keyLimit}
                  setLimit={setKeyLimit}
                />
              </Col>
              <Col numColSpan={1}>
                <RegulationBreakdown violations={userViolations} />
              </Col>
              <Col numColSpan={1}>
                <RequestTypeBreakdown violations={userViolations} />
              </Col>
            </Grid>
          </TabPanel>
          <TabPanel>
            <Grid numItems={2} className="gap-2 w-full">
              <Col numColSpan={2}>
                <EntityBreakdown
                  title="All Keys — Violation Counts"
                  subtitle="Click a key to see individual violations"
                  rows={keyRows}
                  onRowClick={(row) => { setDrawerKey(row.key); setDrawerOpen(true); }}
                  nameColumn="Virtual Key (Team)"
                  limit={keyLimit}
                  setLimit={setKeyLimit}
                />
              </Col>
            </Grid>
          </TabPanel>
        </TabPanels>
      </TabGroup>

      <Drawer
        title={drawerKey ? `Violations: ${drawerKey}` : ""}
        placement="right"
        width={860}
        open={drawerOpen}
        onClose={() => { setDrawerOpen(false); setDrawerKey(null); }}
      >
        {drawerKey && (
          <>
            <div className="mb-4 p-3 bg-gray-50 rounded-lg text-sm">
              <span className="text-gray-500">Owner: </span>
              <span className="font-medium">{MOCK_KEY_OWNERS[drawerKey]?.user}</span>
              <span className="mx-2 text-gray-300">|</span>
              <span className="text-gray-500">Team: </span>
              <span className="font-medium">{MOCK_KEY_OWNERS[drawerKey]?.team}</span>
            </div>
            <ViolationListPanel violations={drawerViolations} emptyMessage="No violations for this key" />
          </>
        )}
      </Drawer>
    </>
  );
};

// ─── Main Component ─────────────────────────────────────────────────────────

const PolicyComplianceTab: React.FC = () => {
  const initialFromDate = useMemo(() => new Date(Date.now() - 7 * 24 * 60 * 60 * 1000), []);
  const initialToDate = useMemo(() => new Date(), []);
  const [dateValue, setDateValue] = useState<DateRangePickerValue>({
    from: initialFromDate,
    to: initialToDate,
  });
  const [complianceView, setComplianceView] = useState<ComplianceView>("global");

  return (
    <div style={{ width: "100%" }} className="relative">
      <div className="flex items-end justify-between gap-6 mb-6">
        <div className="flex-1">
          <div className="flex items-end justify-between gap-6 mb-4 w-full">
            <ComplianceViewSelect value={complianceView} onChange={setComplianceView} />
            <AdvancedDatePicker value={dateValue} onValueChange={setDateValue} />
          </div>

          {complianceView === "global" && <GlobalComplianceView />}
          {complianceView === "team" && <TeamComplianceView />}
          {complianceView === "key" && <KeyComplianceView />}
          {complianceView === "user" && <UserComplianceView />}
        </div>
      </div>
    </div>
  );
};

export default PolicyComplianceTab;
