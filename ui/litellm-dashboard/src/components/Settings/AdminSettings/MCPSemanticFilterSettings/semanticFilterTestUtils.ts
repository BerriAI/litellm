import NotificationManager from "@/components/molecules/notifications_manager";
import { testMCPSemanticFilter } from "@/components/networking";

export interface TestResult {
  totalTools: number;
  selectedTools: number;
  tools: string[];
}

interface FilterHeaders {
  filter: string | null;
  tools: string | null;
}

const parseFilterHeaders = (headers: FilterHeaders): TestResult | null => {
  if (!headers.filter) {
    return null;
  }

  const [total, selected] = headers.filter.split("->").map(Number);
  const tools = headers.tools
    ? headers.tools.split(",").map((name) => name.trim())
    : [];

  return { totalTools: total, selectedTools: selected, tools };
};

export const runSemanticFilterTest = async ({
  accessToken,
  testModel,
  testQuery,
  setIsTesting,
  setTestResult,
}: {
  accessToken: string;
  testModel: string;
  testQuery: string;
  setIsTesting: (value: boolean) => void;
  setTestResult: (result: TestResult | null) => void;
}) => {
  if (!testQuery || !testModel || !accessToken) {
    NotificationManager.error("Please enter a query and select a model");
    return;
  }

  setIsTesting(true);
  setTestResult(null);

  try {
    const { headers } = await testMCPSemanticFilter(
      accessToken,
      testModel,
      testQuery
    );
    const parsedResult = parseFilterHeaders(headers);

    if (!parsedResult) {
      NotificationManager.warning(
        "Semantic filter is not enabled or no tools were filtered"
      );
      return;
    }

    setTestResult(parsedResult);
    NotificationManager.success("Semantic filter test completed successfully");
  } catch (error) {
    console.error("Test failed:", error);
    NotificationManager.error("Failed to test semantic filter");
  } finally {
    setIsTesting(false);
  }
};

export const getCurlCommand = (testModel: string, testQuery: string) =>
  `curl --location 'http://localhost:4000/v1/responses' \\
--header 'Content-Type: application/json' \\
--header 'Authorization: Bearer sk-1234' \\
--data '{
    "model": "${testModel}",
    "input": [
    {
      "role": "user",
      "content": "${testQuery || "Your query here"}",
      "type": "message"
    }
  ],
    "tools": [
        {
            "type": "mcp",
            "server_url": "litellm_proxy",
            "require_approval": "never"
        }
    ],
    "tool_choice": "required"
}'`;
