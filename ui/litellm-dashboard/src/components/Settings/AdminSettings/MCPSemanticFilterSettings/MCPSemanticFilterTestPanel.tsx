import {
  Alert,
  AlertDescription,
  AlertTitle,
} from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from "@/components/ui/tabs";
import { Code, PlayCircle } from "lucide-react";
import ModelSelector from "@/components/common_components/ModelSelector";
import { TestResult } from "./semanticFilterTestUtils";

interface MCPSemanticFilterTestPanelProps {
  accessToken: string | null;
  testQuery: string;
  setTestQuery: (value: string) => void;
  testModel: string;
  setTestModel: (value: string) => void;
  isTesting: boolean;
  onTest: () => void;
  filterEnabled: boolean;
  testResult: TestResult | null;
  curlCommand: string;
}

export default function MCPSemanticFilterTestPanel({
  accessToken,
  testQuery,
  setTestQuery,
  testModel,
  setTestModel,
  isTesting,
  onTest,
  filterEnabled,
  testResult,
  curlCommand,
}: MCPSemanticFilterTestPanelProps) {
  return (
    <Card className="mb-4 p-6">
      <h4 className="text-base font-semibold mb-4">Test Configuration</h4>
      <Tabs defaultValue="test">
        <TabsList>
          <TabsTrigger value="test">Test</TabsTrigger>
          <TabsTrigger value="api">API Usage</TabsTrigger>
        </TabsList>
        <TabsContent value="test">
          <div className="space-y-4">
            <div>
              <Label className="flex items-center gap-1 mb-2">
                <PlayCircle className="h-4 w-4" />
                Test Query
              </Label>
              <Textarea
                placeholder="Enter a test query to see which tools would be selected..."
                value={testQuery}
                onChange={(e) => setTestQuery(e.target.value)}
                rows={4}
                disabled={isTesting}
              />
            </div>

            <div>
              <ModelSelector
                accessToken={accessToken || ""}
                value={testModel}
                onChange={setTestModel}
                disabled={isTesting}
                showLabel={true}
                labelText="Select Model"
              />
            </div>

            <Button
              onClick={onTest}
              disabled={isTesting || !testQuery || !testModel || !filterEnabled}
              className="w-full"
            >
              <PlayCircle className="h-4 w-4" />
              {isTesting ? "Testing…" : "Test Filter"}
            </Button>

            {!filterEnabled && (
              <Alert className="border-amber-300 bg-amber-50 text-amber-900 dark:bg-amber-950/30 dark:text-amber-200">
                <AlertTitle>Semantic filtering is disabled</AlertTitle>
                <AlertDescription>
                  Enable semantic filtering and save settings to test the
                  filter.
                </AlertDescription>
              </Alert>
            )}

            {testResult && (
              <div>
                <h5 className="text-base font-semibold mb-3">Results</h5>
                <Alert className="mb-4 border-emerald-300 bg-emerald-50 text-emerald-900 dark:bg-emerald-950/30 dark:text-emerald-200">
                  <AlertTitle>
                    {testResult.selectedTools} tools selected
                  </AlertTitle>
                  <AlertDescription>
                    Filtered from {testResult.totalTools} available tools
                  </AlertDescription>
                </Alert>
                <div>
                  <span className="font-semibold block mb-2">
                    Selected Tools:
                  </span>
                  <ul className="pl-5 m-0 space-y-1">
                    {testResult.tools.map((tool, index) => (
                      <li key={index}>
                        <span>{tool}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            )}
          </div>
        </TabsContent>
        <TabsContent value="api">
          <div>
            <div className="flex items-center gap-2 mb-2">
              <Code className="h-4 w-4" />
              <span className="font-semibold">API Usage</span>
            </div>
            <p className="text-sm text-muted-foreground mb-2">
              Use this curl command to test the semantic filter with your
              current configuration.
            </p>
            <span className="font-semibold block mb-2">
              Response headers to check:
            </span>
            <ul className="pl-5 mb-3 space-y-2">
              <li>
                <span>
                  x-litellm-semantic-filter: shows total tools → selected tools
                </span>
                <span className="text-muted-foreground block text-xs">
                  Example: 10→3
                </span>
              </li>
              <li>
                <span>
                  x-litellm-semantic-filter-tools: CSV of selected tool names
                </span>
                <span className="text-muted-foreground block text-xs">
                  Example: wikipedia-fetch,github-search,slack-post
                </span>
              </li>
            </ul>
            <pre className="bg-muted border border-border p-3 rounded-md overflow-auto text-xs m-0">
              {curlCommand}
            </pre>
          </div>
        </TabsContent>
      </Tabs>
    </Card>
  );
}
