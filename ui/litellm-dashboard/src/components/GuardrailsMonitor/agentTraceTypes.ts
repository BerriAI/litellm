/**
 * Types and mock data for agent trace demo in GuardrailsMonitor LogViewer.
 * Supports orchestrator → sub-agents → LLM + MCP hierarchy.
 */

export type SpanType =
  | "orchestrator"
  | "agent"
  | "llm"
  | "mcp"
  | "function";

export type SpanStatus = "success" | "error" | "running" | "retry";

export interface Span {
  id: string;
  name: string;
  type: SpanType;
  status: SpanStatus;
  startMs: number;
  durationMs: number;
  model?: string;
  tokens?: { prompt: number; completion: number };
  cost?: number;
  mcpServer?: string;
  mcpTool?: string;
  input?: string;
  output?: string;
  children?: Span[];
}

export interface AgentTraceSession {
  id: string;
  shortId: string;
  rootAgentName: string;
  timestamp: string;
  relativeTime: string;
  totalSpans: number;
  totalCost: number;
  totalDurationMs: number;
  status: SpanStatus;
  spans: Span[];
}

/** First 2 sessions for demo: always shown as first 1–2 rows in LogViewer. */
export const MOCK_AGENT_TRACE_SESSIONS: AgentTraceSession[] = [
  {
    id: "0c4b4759-83aa-4f2e-b91c-d3e7f8a12345",
    shortId: "0c4b4759-83aa",
    rootAgentName: "Currency Research Agent",
    timestamp: "2026-03-02 09:36:12 PM",
    relativeTime: "2 min ago",
    totalSpans: 10,
    totalCost: 0.00366,
    totalDurationMs: 2677,
    status: "success",
    spans: [
      {
        id: "span-orch-001",
        name: "OrchestratorAgent",
        type: "orchestrator",
        status: "success",
        startMs: 0,
        durationMs: 2677,
        input: JSON.stringify(
          {
            task: "Research current currency exchange rates and recent news for USD/EUR, USD/GBP",
            context: { user_id: "u_8821", session: "0c4b4759" },
          },
          null,
          2
        ),
        output: JSON.stringify(
          {
            summary:
              "Completed currency research. EUR/USD at 1.0842 (+0.23%), GBP/USD at 1.2634 (+0.11%). Key news: ECB holds rates, UK inflation data due Thursday.",
            agents_used: ["CurrencyAgent", "NewsAgent"],
            total_tokens: 1026,
          },
          null,
          2
        ),
        children: [
          {
            id: "span-currency-001",
            name: "CurrencyAgent",
            type: "agent",
            status: "success",
            startMs: 50,
            durationMs: 2100,
            input: JSON.stringify(
              {
                task: "Fetch current exchange rates and 7-day history for USD/EUR and USD/GBP",
                tools: ["forex-api"],
              },
              null,
              2
            ),
            output: JSON.stringify(
              {
                rates: { "USD/EUR": 1.0842, "USD/GBP": 1.2634 },
                trend: "bullish",
                confidence: 0.87,
              },
              null,
              2
            ),
            children: [
              {
                id: "span-llm-001",
                name: "gpt-4o",
                type: "llm",
                status: "success",
                startMs: 100,
                durationMs: 1346,
                model: "gpt-4o",
                tokens: { prompt: 198, completion: 74 },
                cost: 0.00096,
                input: JSON.stringify(
                  {
                    messages: [
                      {
                        role: "system",
                        content:
                          "You are a currency research agent. Use the available tools to fetch exchange rates.",
                      },
                      {
                        role: "user",
                        content:
                          "Get current USD/EUR and USD/GBP rates and 7-day history.",
                      },
                    ],
                    tools: [
                      { name: "get_rates", description: "Get current exchange rates" },
                      { name: "get_history", description: "Get historical rate data" },
                    ],
                  },
                  null,
                  2
                ),
                output: JSON.stringify(
                  {
                    role: "assistant",
                    content: null,
                    tool_calls: [
                      {
                        id: "call_abc123",
                        type: "function",
                        function: {
                          name: "get_rates",
                          arguments: '{"pairs": ["USD/EUR", "USD/GBP"]}',
                        },
                      },
                    ],
                  },
                  null,
                  2
                ),
                children: [
                  {
                    id: "span-mcp-001",
                    name: "forex-api/get_rates",
                    type: "mcp",
                    status: "success",
                    startMs: 150,
                    durationMs: 234,
                    mcpServer: "forex-api",
                    mcpTool: "get_rates",
                    input: JSON.stringify(
                      { pairs: ["USD/EUR", "USD/GBP"], timestamp: "latest" },
                      null,
                      2
                    ),
                    output: JSON.stringify(
                      {
                        USD_EUR: 1.0842,
                        USD_GBP: 1.2634,
                        timestamp: "2026-03-02T21:36:12Z",
                        source: "ECB",
                      },
                      null,
                      2
                    ),
                  },
                  {
                    id: "span-mcp-002",
                    name: "forex-api/get_history",
                    type: "mcp",
                    status: "success",
                    startMs: 400,
                    durationMs: 189,
                    mcpServer: "forex-api",
                    mcpTool: "get_history",
                    input: JSON.stringify(
                      { pairs: ["USD/EUR", "USD/GBP"], days: 7 },
                      null,
                      2
                    ),
                    output: JSON.stringify(
                      {
                        data: [
                          { date: "2026-02-24", USD_EUR: 1.0791 },
                          { date: "2026-02-25", USD_EUR: 1.0803 },
                          { date: "2026-02-26", USD_EUR: 1.0821 },
                          { date: "2026-02-27", USD_EUR: 1.0834 },
                          { date: "2026-02-28", USD_EUR: 1.0842 },
                        ],
                      },
                      null,
                      2
                    ),
                  },
                ],
              },
              {
                id: "span-llm-002",
                name: "gpt-4o",
                type: "llm",
                status: "success",
                startMs: 1500,
                durationMs: 826,
                model: "gpt-4o",
                tokens: { prompt: 171, completion: 44 },
                cost: 0.00087,
                input: JSON.stringify(
                  {
                    messages: [
                      {
                        role: "system",
                        content:
                          "Analyze the fetched currency data and provide a summary.",
                      },
                      {
                        role: "user",
                        content: "Summarize the exchange rate data and trends.",
                      },
                      {
                        role: "tool",
                        content: '{"USD_EUR": 1.0842, "USD_GBP": 1.2634}',
                      },
                    ],
                  },
                  null,
                  2
                ),
                output: JSON.stringify(
                  {
                    role: "assistant",
                    content:
                      "EUR/USD is trading at 1.0842, up 0.23% over the past week showing a steady bullish trend. GBP/USD at 1.2634 with modest gains of 0.11%. Both pairs showing positive momentum.",
                  },
                  null,
                  2
                ),
              },
            ],
          },
          {
            id: "span-news-001",
            name: "NewsAgent",
            type: "agent",
            status: "success",
            startMs: 50,
            durationMs: 1800,
            input: JSON.stringify(
              {
                task: "Search for recent news about USD/EUR and USD/GBP exchange rates",
                tools: ["web-search"],
              },
              null,
              2
            ),
            output: JSON.stringify(
              {
                articles: 3,
                sentiment: "neutral",
                key_events: ["ECB rate hold", "UK CPI data pending"],
              },
              null,
              2
            ),
            children: [
              {
                id: "span-llm-003",
                name: "gpt-4o",
                type: "llm",
                status: "success",
                startMs: 100,
                durationMs: 923,
                model: "gpt-4o",
                tokens: { prompt: 245, completion: 96 },
                cost: 0.00112,
                input: JSON.stringify(
                  {
                    messages: [
                      {
                        role: "system",
                        content: "You are a financial news research agent.",
                      },
                      {
                        role: "user",
                        content:
                          "Find recent news about EUR and GBP currency movements.",
                      },
                    ],
                    tools: [
                      { name: "query", description: "Search the web" },
                      { name: "fetch", description: "Fetch a URL" },
                    ],
                  },
                  null,
                  2
                ),
                output: JSON.stringify(
                  {
                    role: "assistant",
                    content: null,
                    tool_calls: [
                      {
                        id: "call_def456",
                        type: "function",
                        function: {
                          name: "query",
                          arguments:
                            '{"q": "EUR USD GBP exchange rate news March 2026"}',
                        },
                      },
                    ],
                  },
                  null,
                  2
                ),
                children: [
                  {
                    id: "span-mcp-003",
                    name: "web-search/query",
                    type: "mcp",
                    status: "success",
                    startMs: 150,
                    durationMs: 445,
                    mcpServer: "web-search",
                    mcpTool: "query",
                    input: JSON.stringify(
                      { q: "EUR USD GBP exchange rate news March 2026", num_results: 5 },
                      null,
                      2
                    ),
                    output: JSON.stringify(
                      {
                        results: [
                          {
                            title: "ECB Holds Rates, Euro Steady Against Dollar",
                            url: "https://reuters.com/...",
                            snippet:
                              "The European Central Bank held rates steady...",
                          },
                          {
                            title: "UK Inflation Data Due Thursday, GBP Cautious",
                            url: "https://ft.com/...",
                            snippet: "Markets await UK CPI data...",
                          },
                        ],
                      },
                      null,
                      2
                    ),
                  },
                ],
              },
              {
                id: "span-llm-004",
                name: "gpt-4o",
                type: "llm",
                status: "success",
                startMs: 1100,
                durationMs: 654,
                model: "gpt-4o",
                tokens: { prompt: 156, completion: 42 },
                cost: 0.00071,
                input: JSON.stringify(
                  {
                    messages: [
                      {
                        role: "system",
                        content: "Summarize the news findings into key insights.",
                      },
                      {
                        role: "tool",
                        content:
                          "ECB holds rates steady. UK CPI data pending Thursday.",
                      },
                    ],
                  },
                  null,
                  2
                ),
                output: JSON.stringify(
                  {
                    role: "assistant",
                    content:
                      "Key findings: (1) ECB held rates steady — neutral for EUR short-term. (2) UK CPI data due Thursday — potential GBP volatility. (3) Overall sentiment: cautiously neutral with slight bullish bias.",
                  },
                  null,
                  2
                ),
              },
            ],
          },
        ],
      },
    ],
  },
  {
    id: "6419f0a4-82ef-4b3c-a921-e8f2d3c45678",
    shortId: "6419f0a4-82ef",
    rootAgentName: "Travel Booking Agent",
    timestamp: "2026-03-02 09:34:27 PM",
    relativeTime: "4 min ago",
    totalSpans: 4,
    totalCost: 0.0031,
    totalDurationMs: 3200,
    status: "success",
    spans: [],
  },
];
