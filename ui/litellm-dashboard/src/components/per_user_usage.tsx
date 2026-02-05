import React, { useState, useEffect } from "react";
import {
  Title,
  Subtitle,
  Table,
  TableHead,
  TableRow,
  TableHeaderCell,
  TableBody,
  TableCell,
  BarChart,
  Text,
  Button,
  Tab,
  TabGroup,
  TabList,
  TabPanel,
  TabPanels,
} from "@tremor/react";
import { perUserAnalyticsCall } from "./networking";

interface PerUserMetrics {
  user_id: string;
  user_email: string | null;
  user_agent: string | null;
  successful_requests: number;
  failed_requests: number;
  total_requests: number;
  total_tokens: number;
  spend: number;
}

interface PerUserAnalyticsResponse {
  results: PerUserMetrics[];
  total_count: number;
  page: number;
  page_size: number;
  total_pages: number;
}

interface PerUserUsageProps {
  accessToken: string | null;
  selectedTags: string[];
  selectedTeam?: string;  // DEMO: Team filter
  selectedAgents?: string[];  // DEMO: Agent filter
  formatAbbreviatedNumber: (value: number, decimalPlaces?: number) => string;
}

// DEMO: Clean agent name mapping for display
const AGENT_DISPLAY_NAMES: Record<string, string> = {
  "claude-code": "Claude Code",
  "claude-code-max": "Claude Code Max",
  "codex-cli": "Codex CLI",
  "GithubCopilot": "GitHub Copilot IDE",
  "go-gh": "GitHub Copilot CLI",
};

// DEMO: Helper function to clean agent names for display
const cleanAgentName = (raw: string | null): string => {
  if (!raw) return "Unknown";
  
  // Strip "User-Agent:" prefix
  let name = raw.replace(/^User-Agent:\s*/i, "");
  
  // Handle versioned strings (e.g., "GithubCopilot/1.155.0")
  if (name.startsWith("GithubCopilot")) return "GitHub Copilot IDE";
  if (name === "go-gh") return "GitHub Copilot CLI";
  if (name.startsWith("Mozilla/")) return "Browser";
  
  return AGENT_DISPLAY_NAMES[name] || name;
};

// DEMO: Helper function to get auth method based on agent name
const getAuthMethod = (agentName: string | null): string => {
  const clean = cleanAgentName(agentName);
  
  if (clean === "Claude Code Max") return "Client OAuth (Anthropic)";
  if (clean === "GitHub Copilot IDE") return "Client OAuth (GitHub)";
  if (clean === "GitHub Copilot CLI") return "Client OAuth (GitHub)";
  
  // Claude Code and Codex CLI use proxy-managed keys
  return "AI Gateway Key";
};

// DEMO: Auth badge component with colored styling and clear distinction
const AuthBadge: React.FC<{ method: string }> = ({ method }) => {
  const isClientOAuth = method.startsWith("Client OAuth");
  
  // Client OAuth = green (user's own credentials, client-side)
  // AI Gateway Key = blue (proxy-managed, server-side)
  const style: React.CSSProperties = {
    display: "inline-block",
    padding: "3px 10px",
    borderRadius: "6px",
    fontSize: "12px",
    fontWeight: 600,
    backgroundColor: isClientOAuth ? "#dcfce7" : "#dbeafe",  // green-100 or blue-100
    color: isClientOAuth ? "#166534" : "#1e40af",              // green-800 or blue-800
    border: isClientOAuth ? "1px solid #86efac" : "1px solid #93c5fd", // green-300 or blue-300
  };
  
  return <span style={style}>{method}</span>;
};

// DEMO: Extended interface for demo users with auth method and team
interface DemoUserMetrics extends PerUserMetrics {
  auth_method: string;
  team: string;
}

// DEMO: User to team mapping
const USER_TEAM_MAP: Record<string, string> = {
  "alice@company.com": "Operations AI",
  "bob@company.com": "Dispatch",
  "charlie@company.com": "Platform",
  "diana@company.com": "Voice",
  "eve@company.com": "Platform",
  "frank@company.com": "Dispatch",
  "grace@company.com": "Voice",
};

// DEMO: Hardcoded per-user data for demo purposes with auth method and team
// - "AI Gateway Key" = credentials managed by LiteLLM proxy (server-side)
// - "Client OAuth (Provider)" = user's own OAuth credentials (client-side, BYOK)
const DEMO_USERS: DemoUserMetrics[] = [
  { user_id: "user-1", user_email: "alice@company.com", team: "Operations AI", auth_method: "AI Gateway Key", user_agent: "Claude Code", successful_requests: 2841, failed_requests: 8, total_requests: 2849, total_tokens: 3800000, spend: 176.20 },
  { user_id: "user-2", user_email: "bob@company.com", team: "Dispatch", auth_method: "AI Gateway Key", user_agent: "Codex CLI", successful_requests: 2103, failed_requests: 7, total_requests: 2110, total_tokens: 2900000, spend: 129.45 },
  { user_id: "user-3", user_email: "charlie@company.com", team: "Platform", auth_method: "Client OAuth (GitHub)", user_agent: "GitHub Copilot IDE", successful_requests: 5612, failed_requests: 3, total_requests: 5615, total_tokens: 6800000, spend: 203.12 },
  { user_id: "user-4", user_email: "diana@company.com", team: "Voice", auth_method: "Client OAuth (Anthropic)", user_agent: "Claude Code Max", successful_requests: 1987, failed_requests: 8, total_requests: 1995, total_tokens: 2400000, spend: 112.80 },
  { user_id: "user-5", user_email: "eve@company.com", team: "Platform", auth_method: "Client OAuth (GitHub)", user_agent: "GitHub Copilot IDE", successful_requests: 4231, failed_requests: 2, total_requests: 4233, total_tokens: 5100000, spend: 152.67 },
  { user_id: "user-6", user_email: "frank@company.com", team: "Dispatch", auth_method: "Client OAuth (GitHub)", user_agent: "GitHub Copilot CLI", successful_requests: 1876, failed_requests: 3, total_requests: 1879, total_tokens: 2600000, spend: 98.34 },
  { user_id: "user-7", user_email: "grace@company.com", team: "Voice", auth_method: "AI Gateway Key", user_agent: "Claude Code", successful_requests: 1542, failed_requests: 2, total_requests: 1544, total_tokens: 2100000, spend: 89.50 },
];

const PerUserUsage: React.FC<PerUserUsageProps> = ({ accessToken, selectedTags, selectedTeam, selectedAgents, formatAbbreviatedNumber }) => {
  // Maximum number of user agent categories to show in charts to prevent color palette overflow
  const MAX_USER_AGENTS = 8;

  // DEMO: Filter users by selected team and selected agents
  const filteredUsers = DEMO_USERS.filter((user) => {
    // Filter by team
    if (selectedTeam && selectedTeam !== "All Teams" && user.team !== selectedTeam) {
      return false;
    }
    // Filter by agents
    if (selectedAgents && selectedAgents.length > 0) {
      const userAgentClean = cleanAgentName(user.user_agent);
      if (!selectedAgents.includes(userAgentClean)) {
        return false;
      }
    }
    return true;
  });
  const [perUserData, setPerUserData] = useState<PerUserAnalyticsResponse>({
    results: [],
    total_count: 0,
    page: 1,
    page_size: 50,
    total_pages: 0,
  });

  const [loading, setLoading] = useState(false);
  const [currentPage, setCurrentPage] = useState(1);

  const fetchPerUserData = async () => {
    if (!accessToken) return;

    setLoading(true);
    try {
      const response = await perUserAnalyticsCall(
        accessToken,
        currentPage,
        50,
        selectedTags.length > 0 ? selectedTags : undefined,
      );
      setPerUserData(response);
    } catch (error) {
      console.error("Failed to fetch per-user data:", error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchPerUserData();
  }, [accessToken, selectedTags, currentPage]);

  const handleNextPage = () => {
    if (currentPage < perUserData.total_pages) {
      setCurrentPage(currentPage + 1);
    }
  };

  const handlePrevPage = () => {
    if (currentPage > 1) {
      setCurrentPage(currentPage - 1);
    }
  };

  return (
    <div className="mb-6">
      <Title>Per User Usage</Title>
      <Subtitle>Individual developer usage metrics</Subtitle>

      <TabGroup>
        <TabList className="mb-6">
          <Tab>User Details</Tab>
          <Tab>Usage Distribution</Tab>
        </TabList>

        <TabPanels>
          {/* DEMO: Tab 1 - User Details Table with hardcoded demo data, Team, and Auth Method columns */}
          <TabPanel>
            <Table>
              <TableHead>
                <TableRow>
                  <TableHeaderCell>User ID</TableHeaderCell>
                  <TableHeaderCell>User Email</TableHeaderCell>
                  <TableHeaderCell>Team</TableHeaderCell>
                  <TableHeaderCell>Auth Method</TableHeaderCell>
                  <TableHeaderCell>User Agent</TableHeaderCell>
                  <TableHeaderCell className="text-right">Success Generations</TableHeaderCell>
                  <TableHeaderCell className="text-right">Total Tokens</TableHeaderCell>
                  <TableHeaderCell className="text-right">Failed Requests</TableHeaderCell>
                  <TableHeaderCell className="text-right">Total Cost</TableHeaderCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {/* DEMO: Using filtered demo users data with team and auth method */}
                {filteredUsers.map((item: DemoUserMetrics, index: number) => (
                  <TableRow key={index}>
                    <TableCell>
                      <Text className="font-medium">{item.user_id}</Text>
                    </TableCell>
                    <TableCell>
                      <Text>{item.user_email || "N/A"}</Text>
                    </TableCell>
                    <TableCell>
                      <Text className="text-gray-600">{item.team}</Text>
                    </TableCell>
                    <TableCell>
                      <AuthBadge method={item.auth_method} />
                    </TableCell>
                    <TableCell>
                      <Text>{cleanAgentName(item.user_agent)}</Text>
                    </TableCell>
                    <TableCell className="text-right">
                      <Text>{formatAbbreviatedNumber(item.successful_requests)}</Text>
                    </TableCell>
                    <TableCell className="text-right">
                      <Text>{formatAbbreviatedNumber(item.total_tokens)}</Text>
                    </TableCell>
                    <TableCell className="text-right">
                      <Text>{formatAbbreviatedNumber(item.failed_requests)}</Text>
                    </TableCell>
                    <TableCell className="text-right">
                      <Text>${formatAbbreviatedNumber(item.spend, 2)}</Text>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TabPanel>

          {/* DEMO: Tab 2 - Usage Distribution Histogram with filtered demo data */}
          <TabPanel>
            <div className="mb-4">
              <Title className="text-lg">User Usage Distribution</Title>
              <Subtitle>Number of users by successful request frequency</Subtitle>
            </div>

            <BarChart
              data={(() => {
                // DEMO: Get unique clean agent names from filtered demo data
                const userAgentCounts = new Map<string, number>();
                filteredUsers.forEach((item: PerUserMetrics) => {
                  const agent = cleanAgentName(item.user_agent);
                  userAgentCounts.set(agent, (userAgentCounts.get(agent) || 0) + 1);
                });

                const topUserAgents = Array.from(userAgentCounts.entries())
                  .sort(([, a], [, b]) => b - a)
                  .slice(0, MAX_USER_AGENTS)
                  .map(([agent]) => agent);

                // Categorize users by successful request count and user agent
                const categories = {
                  "1-9 requests": { range: [1, 9], agents: {} as Record<string, number> },
                  "10-99 requests": { range: [10, 99], agents: {} as Record<string, number> },
                  "100-999 requests": { range: [100, 999], agents: {} as Record<string, number> },
                  "1K-9.9K requests": { range: [1000, 9999], agents: {} as Record<string, number> },
                  "10K-99.9K requests": { range: [10000, 99999], agents: {} as Record<string, number> },
                  "100K+ requests": { range: [100000, Infinity], agents: {} as Record<string, number> },
                };

                // DEMO: Count users in each category by user agent (using filtered data)
                filteredUsers.forEach((item: PerUserMetrics) => {
                  const successCount = item.successful_requests;
                  const userAgent = cleanAgentName(item.user_agent);

                  // Only process if this is one of the top user agents
                  if (topUserAgents.includes(userAgent)) {
                    Object.entries(categories).forEach(([categoryName, category]) => {
                      if (successCount >= category.range[0] && successCount <= category.range[1]) {
                        if (!category.agents[userAgent]) {
                          category.agents[userAgent] = 0;
                        }
                        category.agents[userAgent]++;
                      }
                    });
                  }
                });

                // Convert to chart data format for stacked bar chart
                return Object.entries(categories).map(([categoryName, category]) => {
                  const dataPoint: Record<string, any> = { category: categoryName };

                  // Add count for each top user agent
                  topUserAgents.forEach((agent) => {
                    dataPoint[agent] = category.agents[agent] || 0;
                  });

                  return dataPoint;
                });
              })()}
              index="category"
              categories={(() => {
                // DEMO: Count user agents by frequency and get top ones with clean names (using filtered data)
                const userAgentCounts = new Map<string, number>();
                filteredUsers.forEach((item: PerUserMetrics) => {
                  const agent = cleanAgentName(item.user_agent);
                  userAgentCounts.set(agent, (userAgentCounts.get(agent) || 0) + 1);
                });

                // Sort by frequency (most common first) and limit to top MAX_USER_AGENTS
                return Array.from(userAgentCounts.entries())
                  .sort(([, a], [, b]) => b - a)
                  .slice(0, MAX_USER_AGENTS)
                  .map(([agent]) => agent);
              })()}
              colors={["blue", "green", "orange", "red", "purple", "yellow", "pink", "indigo"]}
              valueFormatter={(value: number) => `${value} users`}
              yAxisWidth={80}
              showLegend={true}
              stack={true}
            />
          </TabPanel>
        </TabPanels>
      </TabGroup>
    </div>
  );
};

export default PerUserUsage;
