// Mock data for MCP approval requests
// This file contains sample data for demonstrating the approval workflow

export interface MCPApprovalRequest {
  approval_id: string;
  server_name: string;
  alias?: string;
  description?: string;
  url: string;
  transport: string;
  auth_type?: string;
  mcp_access_groups: string[];
  allowed_tools: string[];
  status: 'pending' | 'approved' | 'rejected';
  requested_by: string;
  requested_by_team_id?: string;
  team_name?: string;
  requester_email: string;
  created_at: string;
  reviewed_by?: string;
  reviewed_at?: string;
  rejection_reason?: string;
  request_metadata?: {
    business_justification?: string;
    expected_usage?: string;
  };
}

export const mockMCPApprovalRequests: MCPApprovalRequest[] = [
  {
    approval_id: "req-001",
    server_name: "team-engineering-mcp",
    alias: "eng-mcp",
    description: "Engineering team's MCP server for code analysis and automated review",
    url: "https://mcp.engineering.example.com",
    transport: "sse",
    auth_type: "bearer",
    mcp_access_groups: ["engineering", "backend-team"],
    allowed_tools: ["code-review", "lint-check", "test-runner", "coverage-analysis"],
    status: "pending",
    requested_by: "user-123",
    requested_by_team_id: "team-eng",
    team_name: "Engineering Team",
    requester_email: "john.smith@example.com",
    created_at: "2026-02-15T10:30:00Z",
    request_metadata: {
      business_justification: "Need automated code review for PR workflows and continuous integration. Will help reduce manual code review time by ~40%.",
      expected_usage: "~500 requests per day across 50 engineers"
    }
  },
  {
    approval_id: "req-002",
    server_name: "data-science-ml-mcp",
    alias: "ds-ml",
    description: "Data Science team's MCP server for ML model training and evaluation",
    url: "https://mcp.datascience.example.com",
    transport: "sse",
    auth_type: "oauth2",
    mcp_access_groups: ["data-science", "ml-team"],
    allowed_tools: ["train-model", "evaluate-model", "feature-engineering", "data-viz"],
    status: "pending",
    requested_by: "user-456",
    requested_by_team_id: "team-ds",
    team_name: "Data Science Team",
    requester_email: "sarah.jones@example.com",
    created_at: "2026-02-16T14:20:00Z",
    request_metadata: {
      business_justification: "Required for automated ML pipeline. Will enable faster model iteration and A/B testing.",
      expected_usage: "~200 training jobs per week, ~1000 evaluation requests per day"
    }
  },
  {
    approval_id: "req-003",
    server_name: "security-audit-mcp",
    alias: "sec-audit",
    description: "Security team's MCP server for vulnerability scanning and compliance checks",
    url: "https://mcp.security.example.com",
    transport: "sse",
    auth_type: "bearer",
    mcp_access_groups: ["security", "compliance-team"],
    allowed_tools: ["vulnerability-scan", "dependency-check", "secrets-detection", "compliance-audit"],
    status: "pending",
    requested_by: "user-789",
    requested_by_team_id: "team-sec",
    team_name: "Security Team",
    requester_email: "mike.chen@example.com",
    created_at: "2026-02-17T09:15:00Z",
    request_metadata: {
      business_justification: "Critical for SOC 2 compliance. Automated security scanning for all production deployments.",
      expected_usage: "~100 scans per day, triggered on every deployment"
    }
  },
  {
    approval_id: "req-004",
    server_name: "devops-infrastructure-mcp",
    alias: "devops-infra",
    description: "DevOps team's MCP server for infrastructure provisioning and monitoring",
    url: "https://mcp.devops.example.com",
    transport: "stdio",
    auth_type: "api_key",
    mcp_access_groups: ["devops", "sre-team"],
    allowed_tools: ["provision-server", "deploy-service", "health-check", "log-analysis"],
    status: "approved",
    requested_by: "user-111",
    requested_by_team_id: "team-devops",
    team_name: "DevOps Team",
    requester_email: "alex.rodriguez@example.com",
    created_at: "2026-02-10T11:00:00Z",
    reviewed_by: "admin-001",
    reviewed_at: "2026-02-11T15:30:00Z",
    request_metadata: {
      business_justification: "Essential for automated infrastructure management. Reduces manual toil and improves reliability.",
      expected_usage: "~300 infrastructure operations per day"
    }
  },
  {
    approval_id: "req-005",
    server_name: "product-analytics-mcp",
    alias: "prod-analytics",
    description: "Product team's MCP server for user analytics and metrics tracking",
    url: "https://mcp.product.example.com",
    transport: "sse",
    auth_type: "bearer",
    mcp_access_groups: ["product", "analytics-team"],
    allowed_tools: ["track-event", "query-metrics", "generate-report", "create-dashboard"],
    status: "approved",
    requested_by: "user-222",
    requested_by_team_id: "team-product",
    team_name: "Product Team",
    requester_email: "lisa.thompson@example.com",
    created_at: "2026-02-12T13:45:00Z",
    reviewed_by: "admin-002",
    reviewed_at: "2026-02-13T10:00:00Z",
    request_metadata: {
      business_justification: "Required for product analytics and user behavior tracking. Enables data-driven product decisions.",
      expected_usage: "~1000 events per minute, ~50 report generations per day"
    }
  },
  {
    approval_id: "req-006",
    server_name: "marketing-automation-mcp",
    alias: "marketing-auto",
    description: "Marketing team's MCP server for campaign automation and email marketing",
    url: "https://mcp.marketing.example.com",
    transport: "sse",
    auth_type: "oauth2",
    mcp_access_groups: ["marketing"],
    allowed_tools: ["send-campaign", "segment-users", "track-conversion", "ab-test"],
    status: "rejected",
    requested_by: "user-333",
    requested_by_team_id: "team-marketing",
    team_name: "Marketing Team",
    requester_email: "david.kim@example.com",
    created_at: "2026-02-14T16:00:00Z",
    reviewed_by: "admin-001",
    reviewed_at: "2026-02-15T09:20:00Z",
    rejection_reason: "Security concerns: OAuth2 configuration incomplete. Missing required scopes and callback URLs. Please resubmit with proper authentication setup.",
    request_metadata: {
      business_justification: "Need automated marketing campaigns for user engagement and retention.",
      expected_usage: "~10 campaigns per week, ~100K emails per month"
    }
  },
  {
    approval_id: "req-007",
    server_name: "customer-support-mcp",
    alias: "support",
    description: "Customer Support team's MCP server for ticket management and chatbot",
    url: "https://mcp.support.example.com",
    transport: "sse",
    auth_type: "bearer",
    mcp_access_groups: ["support", "customer-success"],
    allowed_tools: ["create-ticket", "update-ticket", "search-knowledge-base", "chatbot-response"],
    status: "pending",
    requested_by: "user-444",
    requested_by_team_id: "team-support",
    team_name: "Customer Support Team",
    requester_email: "emma.wilson@example.com",
    created_at: "2026-02-18T08:00:00Z",
    request_metadata: {
      business_justification: "Enhance customer support response times. Automated ticket routing and AI-powered responses.",
      expected_usage: "~500 tickets per day, ~2000 chatbot interactions per day"
    }
  },
  {
    approval_id: "req-008",
    server_name: "finance-reporting-mcp",
    alias: "fin-reports",
    description: "Finance team's MCP server for financial reporting and analysis",
    url: "https://mcp.finance.example.com",
    transport: "stdio",
    auth_type: "mtls",
    mcp_access_groups: ["finance", "accounting"],
    allowed_tools: ["generate-financial-report", "calculate-metrics", "budget-analysis", "forecast"],
    status: "rejected",
    requested_by: "user-555",
    requested_by_team_id: "team-finance",
    team_name: "Finance Team",
    requester_email: "robert.anderson@example.com",
    created_at: "2026-02-13T10:30:00Z",
    reviewed_by: "admin-002",
    reviewed_at: "2026-02-14T11:45:00Z",
    rejection_reason: "Insufficient access controls. Finance data requires additional encryption and audit logging. Please add detailed access policies and data retention policies.",
    request_metadata: {
      business_justification: "Automate monthly financial reports and forecasting. Reduce manual work by 60%.",
      expected_usage: "~50 reports per month, daily metric calculations"
    }
  },
  {
    approval_id: "req-009",
    server_name: "hr-recruitment-mcp",
    alias: "hr-recruit",
    description: "HR team's MCP server for recruitment automation and candidate screening",
    url: "https://mcp.hr.example.com",
    transport: "sse",
    auth_type: "bearer",
    mcp_access_groups: ["hr", "recruitment-team"],
    allowed_tools: ["parse-resume", "screen-candidate", "schedule-interview", "generate-offer"],
    status: "pending",
    requested_by: "user-666",
    requested_by_team_id: "team-hr",
    team_name: "HR Team",
    requester_email: "jessica.brown@example.com",
    created_at: "2026-02-17T15:30:00Z",
    request_metadata: {
      business_justification: "Streamline recruitment process. Automated resume screening and interview scheduling to reduce time-to-hire.",
      expected_usage: "~200 applications per week, ~50 interviews per month"
    }
  },
  {
    approval_id: "req-010",
    server_name: "legal-contract-mcp",
    alias: "legal-docs",
    description: "Legal team's MCP server for contract management and document review",
    url: "https://mcp.legal.example.com",
    transport: "sse",
    auth_type: "mtls",
    mcp_access_groups: ["legal"],
    allowed_tools: ["review-contract", "extract-terms", "compliance-check", "redline-document"],
    status: "approved",
    requested_by: "user-777",
    requested_by_team_id": "team-legal",
    team_name: "Legal Team",
    requester_email: "william.martinez@example.com",
    created_at: "2026-02-11T09:00:00Z",
    reviewed_by: "admin-001",
    reviewed_at: "2026-02-12T14:15:00Z",
    request_metadata: {
      business_justification: "Automated contract review and compliance checking. Speeds up legal review process by 50%.",
      expected_usage: "~30 contracts per week, ongoing compliance monitoring"
    }
  }
];
