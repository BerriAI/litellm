import React, { useState, useEffect } from "react";
import {
  Card,
  Title,
  Text,
  Grid,
  BarChart,
  Metric,
  Subtitle,
  Tab,
  TabGroup,
  TabList,
  TabPanel,
  TabPanels,
  Table,
  TableHead,
  TableRow,
  TableHeaderCell,
  TableBody,
  TableCell,
} from "@tremor/react";
import { Select } from "antd";
import { userAgentSummaryCall, tagDauCall, tagWauCall, tagMauCall, tagDistinctCall } from "./networking";
import PerUserUsage from "./per_user_usage";
import { DateRangePickerValue } from "@tremor/react";
import { ChartLoader } from "./shared/chart_loader";

// New interfaces for the updated API response
interface TagActiveUsersResponse {
  tag: string;
  active_users: number;
  date: string;
  period_start?: string;
  period_end?: string;
}

interface ActiveUsersAnalyticsResponse {
  results: TagActiveUsersResponse[];
}

interface TagSummaryMetrics {
  tag: string;
  unique_users: number;
  total_requests: number;
  successful_requests: number;
  failed_requests: number;
  total_tokens: number;
  total_spend: number;
}

interface TagSummaryResponse {
  results: TagSummaryMetrics[];
}

interface DistinctTagResponse {
  tag: string;
}

interface DistinctTagsResponse {
  results: DistinctTagResponse[];
}

interface UserAgentActivityProps {
  accessToken: string | null;
  userRole: string | null;
  dateValue: DateRangePickerValue;
  onDateChange?: (value: DateRangePickerValue) => void; // Optional - not used anymore
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
const cleanAgentName = (raw: string): string => {
  // Strip "User-Agent:" prefix
  let name = raw.replace(/^User-Agent:\s*/i, "");
  
  // Handle versioned strings (e.g., "GithubCopilot/1.155.0")
  if (name.startsWith("GithubCopilot")) return "GitHub Copilot IDE";
  if (name === "go-gh") return "GitHub Copilot CLI";
  if (name.startsWith("Mozilla/")) return "Browser";
  
  return AGENT_DISPLAY_NAMES[name] || name;
};

// DEMO: Hardcoded agent summary data for demo purposes
const DEMO_AGENTS = [
  {
    name: "GitHub Copilot IDE",
    raw_ua: "GithubCopilot/1.155.0",
    success_requests: 34219,
    total_tokens: 42100000,
    total_cost: 1247.92,
  },
  {
    name: "Claude Code",
    raw_ua: "claude-code",
    success_requests: 12847,
    total_tokens: 18200000,
    total_cost: 842.30,
  },
  {
    name: "Claude Code Max",
    raw_ua: "claude-code-max",
    success_requests: 8421,
    total_tokens: 11700000,
    total_cost: 523.18,
  },
  {
    name: "Codex CLI",
    raw_ua: "codex-cli",
    success_requests: 5102,
    total_tokens: 7100000,
    total_cost: 312.45,
  },
  {
    name: "GitHub Copilot CLI",
    raw_ua: "go-gh",
    success_requests: 2340,
    total_tokens: 3200000,
    total_cost: 98.67,
  },
];

// DEMO: Hardcoded DAU data for demo purposes (7 days with realistic weekday/weekend pattern)
const DEMO_DAU = [
  { date: "2026-01-30", "GitHub Copilot IDE": 18, "Claude Code": 12, "Claude Code Max": 6, "Codex CLI": 4, "GitHub Copilot CLI": 3 },
  { date: "2026-01-31", "GitHub Copilot IDE": 20, "Claude Code": 14, "Claude Code Max": 7, "Codex CLI": 5, "GitHub Copilot CLI": 3 },
  { date: "2026-02-01", "GitHub Copilot IDE": 15, "Claude Code": 10, "Claude Code Max": 5, "Codex CLI": 3, "GitHub Copilot CLI": 2 },
  { date: "2026-02-02", "GitHub Copilot IDE": 14, "Claude Code": 9,  "Claude Code Max": 4, "Codex CLI": 3, "GitHub Copilot CLI": 2 },
  { date: "2026-02-03", "GitHub Copilot IDE": 22, "Claude Code": 16, "Claude Code Max": 8, "Codex CLI": 6, "GitHub Copilot CLI": 4 },
  { date: "2026-02-04", "GitHub Copilot IDE": 24, "Claude Code": 18, "Claude Code Max": 9, "Codex CLI": 7, "GitHub Copilot CLI": 5 },
  { date: "2026-02-05", "GitHub Copilot IDE": 26, "Claude Code": 19, "Claude Code Max": 10, "Codex CLI": 8, "GitHub Copilot CLI": 5 },
];

// DEMO: Agent names for chart categories (excludes Browser/Mozilla)
const DEMO_AGENT_CATEGORIES = ["GitHub Copilot IDE", "Claude Code", "Claude Code Max", "Codex CLI", "GitHub Copilot CLI"];

// DEMO: Available teams for filtering
const DEMO_TEAMS_LIST = ["All Teams", "Operations AI", "Platform", "Voice", "Dispatch"];

// DEMO: Budget data per agent (monthly budget and current usage)
const DEMO_BUDGETS: Record<string, { budget: number; used: number }> = {
  "GitHub Copilot IDE": { budget: 2000, used: 1248 },
  "Claude Code": { budget: 1500, used: 842 },
  "Claude Code Max": { budget: 1000, used: 523 },
  "Codex CLI": { budget: 500, used: 312 },
  "GitHub Copilot CLI": { budget: 200, used: 99 },
};

// DEMO: Guardrails enabled per agent
const DEMO_GUARDRAILS: Record<string, string[]> = {
  "GitHub Copilot IDE": ["PII Redaction", "IP Protection"],
  "Claude Code": ["PII Redaction", "Prompt Injection"],
  "Claude Code Max": ["PII Redaction", "Prompt Injection"],
  "Codex CLI": ["PII Redaction"],
  "GitHub Copilot CLI": ["PII Redaction"],
};

// DEMO: User data for computing team stats (mirrors per_user_usage.tsx)
const DEMO_USERS_FOR_TEAMS = [
  { user_email: "alice@company.com", team: "Operations AI", user_agent: "Claude Code", successful_requests: 2841, total_tokens: 3800000, spend: 176.20 },
  { user_email: "bob@company.com", team: "Dispatch", user_agent: "Codex CLI", successful_requests: 2103, total_tokens: 2900000, spend: 129.45 },
  { user_email: "charlie@company.com", team: "Platform", user_agent: "GitHub Copilot IDE", successful_requests: 5612, total_tokens: 6800000, spend: 203.12 },
  { user_email: "diana@company.com", team: "Voice", user_agent: "Claude Code Max", successful_requests: 1987, total_tokens: 2400000, spend: 112.80 },
  { user_email: "eve@company.com", team: "Platform", user_agent: "GitHub Copilot IDE", successful_requests: 4231, total_tokens: 5100000, spend: 152.67 },
  { user_email: "frank@company.com", team: "Dispatch", user_agent: "GitHub Copilot CLI", successful_requests: 1876, total_tokens: 2600000, spend: 98.34 },
  { user_email: "grace@company.com", team: "Voice", user_agent: "Claude Code", successful_requests: 1542, total_tokens: 2100000, spend: 89.50 },
];

// DEMO: Function to compute team stats from filtered user data
const computeTeamStats = (users: typeof DEMO_USERS_FOR_TEAMS, selectedAgents: string[]) => {
  // Filter users by selected agents
  const filteredUsers = users.filter((user) => {
    if (selectedAgents.length === 0) return true;
    return selectedAgents.includes(user.user_agent);
  });

  // Group by team
  const teamMap = new Map<string, { active_users: number; total_requests: number; total_tokens: number; total_cost: number }>();
  
  filteredUsers.forEach((user) => {
    const existing = teamMap.get(user.team) || { active_users: 0, total_requests: 0, total_tokens: 0, total_cost: 0 };
    teamMap.set(user.team, {
      active_users: existing.active_users + 1,
      total_requests: existing.total_requests + user.successful_requests,
      total_tokens: existing.total_tokens + user.total_tokens,
      total_cost: existing.total_cost + user.spend,
    });
  });

  // Convert to array and sort by total_requests
  return Array.from(teamMap.entries())
    .map(([team, stats]) => ({ team, ...stats }))
    .sort((a, b) => b.total_requests - a.total_requests);
};

// DEMO: Budget progress bar component
const BudgetProgressBar: React.FC<{ budget: number; used: number }> = ({ budget, used }) => {
  const percentage = Math.round((used / budget) * 100);
  
  // Color based on percentage: green < 60%, amber 60-80%, red > 80%
  let fillColor = "#22c55e"; // green
  if (percentage >= 80) {
    fillColor = "#ef4444"; // red
  } else if (percentage >= 60) {
    fillColor = "#eab308"; // amber
  }
  
  return (
    <div className="mt-3">
      <div className="flex justify-between items-center mb-1">
        <Text className="text-xs text-gray-500">Budget: ${budget.toLocaleString()}/mo</Text>
        <Text className="text-xs text-gray-500">{percentage}%</Text>
      </div>
      <div 
        style={{ 
          height: "6px", 
          borderRadius: "3px", 
          backgroundColor: "#e5e7eb",
          overflow: "hidden"
        }}
      >
        <div 
          style={{ 
            height: "100%", 
            width: `${percentage}%`, 
            backgroundColor: fillColor,
            borderRadius: "3px",
            transition: "width 0.3s ease"
          }} 
        />
      </div>
    </div>
  );
};

// DEMO: Guardrails line component
const GuardrailsLine: React.FC<{ guardrails: string[] }> = ({ guardrails }) => {
  if (!guardrails || guardrails.length === 0) return null;
  
  return (
    <div 
      style={{ 
        borderTop: "1px solid #e5e7eb", 
        paddingTop: "8px", 
        marginTop: "8px" 
      }}
    >
      <Text className="text-xs text-gray-500">
        <span style={{ color: "#22c55e", marginRight: "4px" }}>✓</span>
        {guardrails.join(", ")}
      </Text>
    </div>
  );
};

const UserAgentActivity: React.FC<UserAgentActivityProps> = ({ accessToken, userRole, dateValue, onDateChange }) => {
  // Maximum number of categories to show in charts to prevent color palette overflow
  const MAX_CATEGORIES = 10;

  // Separate state for each endpoint
  const [dauData, setDauData] = useState<ActiveUsersAnalyticsResponse>({ results: [] });
  const [wauData, setWauData] = useState<ActiveUsersAnalyticsResponse>({ results: [] });
  const [mauData, setMauData] = useState<ActiveUsersAnalyticsResponse>({ results: [] });
  const [summaryData, setSummaryData] = useState<TagSummaryResponse>({ results: [] });

  const [userAgentFilter, setUserAgentFilter] = useState<string>("");

  // Tag filtering state
  const [availableTags, setAvailableTags] = useState<string[]>([]);
  const [selectedTags, setSelectedTags] = useState<string[]>([]);
  const [tagsLoading, setTagsLoading] = useState(false);

  // DEMO: Team filter state
  const [selectedTeam, setSelectedTeam] = useState<string>("All Teams");

  // DEMO: Agent filter state (for filtering by demo agent names)
  const [selectedAgents, setSelectedAgents] = useState<string[]>([]);

  // Separate loading states for each endpoint
  const [dauLoading, setDauLoading] = useState(false);
  const [wauLoading, setWauLoading] = useState(false);
  const [mauLoading, setMauLoading] = useState(false);
  const [summaryLoading, setSummaryLoading] = useState(false);

  // Use today's date as the end date for all API calls
  const today = new Date();

  const fetchAvailableTags = async () => {
    if (!accessToken) return;

    setTagsLoading(true);
    try {
      const data = await tagDistinctCall(accessToken);
      setAvailableTags(data.results.map((item: DistinctTagResponse) => item.tag));
    } catch (error) {
      console.error("Failed to fetch available tags:", error);
    } finally {
      setTagsLoading(false);
    }
  };

  const fetchDauData = async () => {
    if (!accessToken) return;

    setDauLoading(true);
    try {
      const data = await tagDauCall(
        accessToken,
        today,
        userAgentFilter || undefined,
        selectedTags.length > 0 ? selectedTags : undefined,
      );
      setDauData(data);
    } catch (error) {
      console.error("Failed to fetch DAU data:", error);
    } finally {
      setDauLoading(false);
    }
  };

  const fetchWauData = async () => {
    if (!accessToken) return;

    setWauLoading(true);
    try {
      const data = await tagWauCall(
        accessToken,
        today,
        userAgentFilter || undefined,
        selectedTags.length > 0 ? selectedTags : undefined,
      );
      setWauData(data);
    } catch (error) {
      console.error("Failed to fetch WAU data:", error);
    } finally {
      setWauLoading(false);
    }
  };

  const fetchMauData = async () => {
    if (!accessToken) return;

    setMauLoading(true);
    try {
      const data = await tagMauCall(
        accessToken,
        today,
        userAgentFilter || undefined,
        selectedTags.length > 0 ? selectedTags : undefined,
      );
      setMauData(data);
    } catch (error) {
      console.error("Failed to fetch MAU data:", error);
    } finally {
      setMauLoading(false);
    }
  };

  const fetchSummaryData = async () => {
    if (!accessToken || !dateValue.from || !dateValue.to) return;

    setSummaryLoading(true);
    try {
      const summary = await userAgentSummaryCall(
        accessToken,
        dateValue.from,
        dateValue.to,
        selectedTags.length > 0 ? selectedTags : undefined,
      );
      setSummaryData(summary);
    } catch (error) {
      console.error("Failed to fetch user agent summary data:", error);
    } finally {
      setSummaryLoading(false);
    }
  };

  // Effect to fetch available tags on mount
  useEffect(() => {
    fetchAvailableTags();
  }, [accessToken]);

  // Effect for DAU/WAU/MAU data (independent of date picker)
  useEffect(() => {
    if (!accessToken) return;

    const timeoutId = setTimeout(() => {
      fetchDauData();
      fetchWauData();
      fetchMauData();
    }, 50);

    return () => clearTimeout(timeoutId);
  }, [accessToken, userAgentFilter, selectedTags]);

  // Effect for summary data (depends on date picker)
  useEffect(() => {
    if (!dateValue.from || !dateValue.to) return;

    const timeoutId = setTimeout(() => {
      fetchSummaryData();
    }, 50);

    return () => clearTimeout(timeoutId);
  }, [accessToken, dateValue, selectedTags]);

  // Helper function to extract user agent from tag and clean it for display
  const extractUserAgent = (tag: string): string => {
    let name = tag;
    if (tag.startsWith("User-Agent: ")) {
      name = tag.replace("User-Agent: ", "");
    }
    // Apply clean agent name mapping
    return cleanAgentName(name);
  };

  // Get all user agents for each chart type based on their specific data
  const getAllTagsForData = (data: TagActiveUsersResponse[]) => {
    // Aggregate total active users per tag
    const tagTotals = data.reduce(
      (acc, item) => {
        acc[item.tag] = (acc[item.tag] || 0) + item.active_users;
        return acc;
      },
      {} as Record<string, number>,
    );

    // Sort by total active users and return all tags
    return Object.entries(tagTotals)
      .sort(([, a], [, b]) => b - a)
      .map(([tag]) => tag);
  };

  const allDauTags = getAllTagsForData(dauData.results).slice(0, MAX_CATEGORIES);
  const allWauTags = getAllTagsForData(wauData.results).slice(0, MAX_CATEGORIES);
  const allMauTags = getAllTagsForData(mauData.results).slice(0, MAX_CATEGORIES);

  // Prepare daily chart data (DAU) - always show last 7 days
  const generateDailyChartData = () => {
    const chartData: any[] = [];
    const endDate = new Date();

    // Generate all 7 days
    for (let i = 6; i >= 0; i--) {
      const date = new Date(endDate);
      date.setDate(date.getDate() - i);
      const dateStr = date.toISOString().split("T")[0]; // YYYY-MM-DD format

      const dayEntry: any = { date: dateStr };

      // Initialize all user agents to 0
      allDauTags.forEach((tag) => {
        const userAgent = extractUserAgent(tag);
        dayEntry[userAgent] = 0;
      });

      chartData.push(dayEntry);
    }

    // Fill in actual data
    dauData.results.forEach((item) => {
      const userAgent = extractUserAgent(item.tag);
      const dayEntry = chartData.find((d) => d.date === item.date);
      if (dayEntry) {
        dayEntry[userAgent] = item.active_users;
      }
    });

    return chartData;
  };

  const dailyChartData = generateDailyChartData();

  // Prepare weekly chart data (WAU) - always show all 7 weeks
  const generateWeeklyChartData = () => {
    const chartData: any[] = [];

    // Generate all 7 weeks (Week 1 through Week 7)
    for (let weekNum = 1; weekNum <= 7; weekNum++) {
      const weekEntry: any = { week: `Week ${weekNum}` };

      // Initialize all user agents to 0
      allWauTags.forEach((tag) => {
        const userAgent = extractUserAgent(tag);
        weekEntry[userAgent] = 0;
      });

      chartData.push(weekEntry);
    }

    // Fill in actual data
    wauData.results.forEach((item) => {
      const userAgent = extractUserAgent(item.tag);
      // Extract week number from the date field (e.g., "Week 1 (Jul 27)" -> "Week 1")
      const weekMatch = item.date.match(/Week (\d+)/);
      if (weekMatch) {
        const weekLabel = `Week ${weekMatch[1]}`;
        const weekEntry = chartData.find((d) => d.week === weekLabel);
        if (weekEntry) {
          weekEntry[userAgent] = item.active_users;
        }
      }
    });

    return chartData;
  };

  const weeklyChartData = generateWeeklyChartData();

  // Prepare monthly chart data (MAU) - always show all 7 months
  const generateMonthlyChartData = () => {
    const chartData: any[] = [];

    // Generate all 7 months (Month 1 through Month 7)
    for (let monthNum = 1; monthNum <= 7; monthNum++) {
      const monthEntry: any = { month: `Month ${monthNum}` };

      // Initialize all user agents to 0
      allMauTags.forEach((tag) => {
        const userAgent = extractUserAgent(tag);
        monthEntry[userAgent] = 0;
      });

      chartData.push(monthEntry);
    }

    // Fill in actual data
    mauData.results.forEach((item) => {
      const userAgent = extractUserAgent(item.tag);
      // Extract month number from the date field (e.g., "Month 1 (Jul)" -> "Month 1")
      const monthMatch = item.date.match(/Month (\d+)/);
      if (monthMatch) {
        const monthLabel = `Month ${monthMatch[1]}`;
        const monthEntry = chartData.find((d) => d.month === monthLabel);
        if (monthEntry) {
          monthEntry[userAgent] = item.active_users;
        }
      }
    });

    return chartData;
  };

  const monthlyChartData = generateMonthlyChartData();

  // Format numbers with K, M abbreviations
  const formatAbbreviatedNumber = (value: number, decimalPlaces: number = 0): string => {
    if (value >= 100000000) {
      return (value / 1000000).toFixed(decimalPlaces) + "M";
    } else if (value >= 10000000) {
      return (value / 1000000).toFixed(decimalPlaces) + "M";
    } else if (value >= 1000000) {
      return (value / 1000000).toFixed(decimalPlaces) + "M";
    } else if (value >= 10000) {
      return (value / 1000).toFixed(decimalPlaces) + "K";
    } else if (value >= 1000) {
      return (value / 1000).toFixed(decimalPlaces) + "K";
    } else {
      return value.toFixed(decimalPlaces);
    }
  };

  return (
    <div className="space-y-6 mt-6">
      {/* Summary Section Card */}
      <Card>
        <div className="space-y-6">
          <div className="flex justify-between items-start">
            <div>
              <Title>Summary by User Agent</Title>
              <Subtitle>Performance metrics for different user agents</Subtitle>
            </div>

            {/* DEMO: Filter dropdowns - Team and User Agent */}
            <div className="flex gap-4">
              {/* Team Filter */}
              <div className="w-48">
                <Text className="text-sm font-medium block mb-2">Filter by Team</Text>
                <Select
                  placeholder="All Teams"
                  value={selectedTeam}
                  onChange={setSelectedTeam}
                  style={{ width: "100%" }}
                  className="rounded-md"
                >
                  {DEMO_TEAMS_LIST.map((team) => (
                    <Select.Option key={team} value={team}>
                      {team}
                    </Select.Option>
                  ))}
                </Select>
              </div>

              {/* DEMO: User Agent Filter - Using hardcoded demo agents */}
              <div className="w-64">
                <Text className="text-sm font-medium block mb-2">Filter by User Agents</Text>
                <Select
                  mode="multiple"
                  placeholder="All User Agents"
                  value={selectedAgents}
                  onChange={setSelectedAgents}
                  style={{ width: "100%" }}
                  showSearch={true}
                  allowClear={true}
                  optionFilterProp="label"
                  className="rounded-md"
                  maxTagCount="responsive"
                >
                  {DEMO_AGENT_CATEGORIES.map((agent) => (
                    <Select.Option key={agent} value={agent} label={agent}>
                      {agent}
                    </Select.Option>
                  ))}
                </Select>
              </div>
            </div>
          </div>

          {/* Date Range Picker is controlled by parent component */}

          {/* DEMO: Top 5 User Agents Cards - Using hardcoded demo data */}
          {summaryLoading ? (
            <ChartLoader isDateChanging={false} />
          ) : (
            <Grid numItems={5} className="gap-4">
              {DEMO_AGENTS.map((agent, index) => {
                return (
                  <Card key={index}>
                    <Title>{agent.name}</Title>
                    <div className="mt-4 space-y-3">
                      <div>
                        <Text className="text-sm text-gray-600">Success Requests</Text>
                        <Metric className="text-lg">{formatAbbreviatedNumber(agent.success_requests)}</Metric>
                      </div>
                      <div>
                        <Text className="text-sm text-gray-600">Total Tokens</Text>
                        <Metric className="text-lg">{formatAbbreviatedNumber(agent.total_tokens)}</Metric>
                      </div>
                      <div>
                        <Text className="text-sm text-gray-600">Total Cost</Text>
                        <Metric className="text-lg">${formatAbbreviatedNumber(agent.total_cost, 2)}</Metric>
                      </div>
                    </div>
                  </Card>
                );
              })}
            </Grid>
          )}
        </div>
      </Card>

      {/* DEMO: Main TabGroup - Per User Usage first, Per Team Usage second, then DAU/WAU/MAU */}
      <Card>
        <TabGroup>
          <TabList className="mb-6">
            <Tab>Per User Usage</Tab>
            <Tab>Per Team Usage</Tab>
            <Tab>DAU/WAU/MAU</Tab>
          </TabList>

          <TabPanels>
            {/* Per User Usage Tab Panel - First */}
            <TabPanel>
              <PerUserUsage
                accessToken={accessToken}
                selectedTags={selectedTags}
                selectedTeam={selectedTeam}
                selectedAgents={selectedAgents}
                formatAbbreviatedNumber={formatAbbreviatedNumber}
              />
            </TabPanel>

            {/* DEMO: Per Team Usage Tab Panel - Second (filtered by selected agents) */}
            <TabPanel>
              <div className="mb-6">
                <Title>Per Team Usage</Title>
                <Subtitle>Usage metrics grouped by team</Subtitle>
              </div>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableHeaderCell>Team</TableHeaderCell>
                    <TableHeaderCell className="text-right">Active Users</TableHeaderCell>
                    <TableHeaderCell className="text-right">Total Requests</TableHeaderCell>
                    <TableHeaderCell className="text-right">Total Tokens</TableHeaderCell>
                    <TableHeaderCell className="text-right">Total Cost</TableHeaderCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {computeTeamStats(DEMO_USERS_FOR_TEAMS, selectedAgents).map((team, index) => (
                    <TableRow 
                      key={index} 
                      className="cursor-pointer hover:bg-gray-50"
                      onClick={() => setSelectedTeam(team.team)}
                    >
                      <TableCell>
                        <Text className="font-medium">{team.team}</Text>
                      </TableCell>
                      <TableCell className="text-right">
                        <Text>{team.active_users}</Text>
                      </TableCell>
                      <TableCell className="text-right">
                        <Text>{formatAbbreviatedNumber(team.total_requests)}</Text>
                      </TableCell>
                      <TableCell className="text-right">
                        <Text>{formatAbbreviatedNumber(team.total_tokens)}</Text>
                      </TableCell>
                      <TableCell className="text-right">
                        <Text>${formatAbbreviatedNumber(team.total_cost, 2)}</Text>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
              <Text className="text-xs text-gray-500 mt-4">Click a team row to filter the page by that team</Text>
            </TabPanel>

            {/* DAU/WAU/MAU Tab Panel - Third */}
            <TabPanel>
              <div className="mb-6">
                <Title>DAU, WAU & MAU per Agent</Title>
                <Subtitle>Active users across different time periods</Subtitle>
              </div>

              <TabGroup>
                <TabList className="mb-6">
                  <Tab>DAU</Tab>
                  <Tab>WAU</Tab>
                  <Tab>MAU</Tab>
                </TabList>

                <TabPanels>
                  <TabPanel>
                    <div className="mb-4">
                      <Title className="text-lg">Daily Active Users - Last 7 Days</Title>
                    </div>
                    {/* DEMO: Using hardcoded DAU data instead of API data */}
                    <BarChart
                      data={DEMO_DAU}
                      index="date"
                      categories={DEMO_AGENT_CATEGORIES}
                      colors={["blue", "purple", "violet", "cyan", "green"]}
                      valueFormatter={(value: number) => formatAbbreviatedNumber(value)}
                      yAxisWidth={60}
                      showLegend={true}
                      stack={true}
                    />
                  </TabPanel>

                  <TabPanel>
                    <div className="mb-4">
                      <Title className="text-lg">Weekly Active Users - Last 7 Weeks</Title>
                    </div>
                    {wauLoading ? (
                      <ChartLoader isDateChanging={false} />
                    ) : (
                      <BarChart
                        data={weeklyChartData}
                        index="week"
                        categories={allWauTags.map(extractUserAgent)}
                        valueFormatter={(value: number) => formatAbbreviatedNumber(value)}
                        yAxisWidth={60}
                        showLegend={true}
                        stack={true}
                      />
                    )}
                  </TabPanel>

                  <TabPanel>
                    <div className="mb-4">
                      <Title className="text-lg">Monthly Active Users - Last 7 Months</Title>
                    </div>
                    {mauLoading ? (
                      <ChartLoader isDateChanging={false} />
                    ) : (
                      <BarChart
                        data={monthlyChartData}
                        index="month"
                        categories={allMauTags.map(extractUserAgent)}
                        valueFormatter={(value: number) => formatAbbreviatedNumber(value)}
                        yAxisWidth={60}
                        showLegend={true}
                        stack={true}
                      />
                    )}
                  </TabPanel>
                </TabPanels>
              </TabGroup>
            </TabPanel>
          </TabPanels>
        </TabGroup>
      </Card>
    </div>
  );
};

export default UserAgentActivity;
