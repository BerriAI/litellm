import React, { useState } from "react";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { FlaskConical, Loader2, Search } from "lucide-react";
import { cn } from "@/lib/utils";
import GuardrailTestPanel from "./GuardrailTestPanel";
import { applyGuardrail } from "../networking";
import NotificationsManager from "../molecules/notifications_manager";

interface GuardrailItem {
  guardrail_id?: string;
  guardrail_name: string | null;
  litellm_params: {
    guardrail: string;
    mode: string;
    default_on: boolean;
  };
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  guardrail_info: Record<string, any> | null;
  created_at?: string;
  updated_at?: string;
}

interface GuardrailTestPlaygroundProps {
  guardrailsList: GuardrailItem[];
  isLoading: boolean;
  accessToken: string | null;
  onClose: () => void;
}

interface TestResult {
  guardrailName: string;
  response_text: string;
  latency: number;
}

interface TestError {
  guardrailName: string;
  error: Error;
  latency: number;
}

const GuardrailTestPlayground: React.FC<GuardrailTestPlaygroundProps> = ({
  guardrailsList,
  isLoading,
  accessToken,
}) => {
  const [selectedGuardrails, setSelectedGuardrails] = useState<Set<string>>(
    new Set(),
  );
  const [searchQuery, setSearchQuery] = useState("");
  const [testResults, setTestResults] = useState<TestResult[]>([]);
  const [testErrors, setTestErrors] = useState<TestError[]>([]);
  const [isTesting, setIsTesting] = useState(false);

  const filteredGuardrails = guardrailsList.filter((guardrail) =>
    guardrail.guardrail_name
      ?.toLowerCase()
      .includes(searchQuery.toLowerCase()),
  );

  const toggleGuardrailSelection = (guardrailName: string) => {
    const newSelection = new Set(selectedGuardrails);
    if (newSelection.has(guardrailName)) {
      newSelection.delete(guardrailName);
    } else {
      newSelection.add(guardrailName);
    }
    setSelectedGuardrails(newSelection);
  };

  const handleTestGuardrails = async (text: string) => {
    if (selectedGuardrails.size === 0 || !accessToken) {
      return;
    }

    setIsTesting(true);
    setTestResults([]);
    setTestErrors([]);

    const results: TestResult[] = [];
    const errors: TestError[] = [];

    await Promise.all(
      Array.from(selectedGuardrails).map(async (guardrailName) => {
        const startTime = Date.now();
        try {
          const result = await applyGuardrail(
            accessToken,
            guardrailName,
            text,
            null,
            null,
          );
          const latency = Date.now() - startTime;
          results.push({
            guardrailName,
            response_text: result.response_text,
            latency,
          });
        } catch (error) {
          const latency = Date.now() - startTime;
          console.error(`Error testing guardrail ${guardrailName}:`, error);
          errors.push({
            guardrailName,
            error: error as Error,
            latency,
          });
        }
      }),
    );

    setTestResults(results);
    setTestErrors(errors);
    setIsTesting(false);

    if (results.length > 0) {
      NotificationsManager.success(
        `${results.length} guardrail${results.length > 1 ? "s" : ""} applied successfully`,
      );
    }
    if (errors.length > 0) {
      NotificationsManager.fromBackend(
        `${errors.length} guardrail${errors.length > 1 ? "s" : ""} failed`,
      );
    }
  };

  return (
    <div className="w-full h-[calc(100vh-200px)]">
      <Card className="h-full p-0 overflow-hidden">
        <div className="flex h-full">
          <div className="w-1/4 border-r border-border flex flex-col overflow-hidden">
            <div className="p-4 border-b border-border">
              <div className="mb-3">
                <h3 className="text-lg font-semibold mb-3">Guardrails</h3>
                <div className="relative">
                  <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground pointer-events-none" />
                  <Input
                    placeholder="Search guardrails..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="pl-8"
                  />
                </div>
              </div>
            </div>

            <div className="flex-1 overflow-auto">
              {isLoading ? (
                <div className="flex items-center justify-center h-32">
                  <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
                </div>
              ) : filteredGuardrails.length === 0 ? (
                <div className="p-4 text-center text-sm text-muted-foreground">
                  {searchQuery
                    ? "No guardrails match your search"
                    : "No guardrails available"}
                </div>
              ) : (
                <ul className="divide-y divide-border">
                  {filteredGuardrails.map((guardrail) => {
                    const isSelected = selectedGuardrails.has(
                      guardrail.guardrail_name || "",
                    );
                    return (
                      <li
                        key={guardrail.guardrail_id || guardrail.guardrail_name}
                        onClick={() => {
                          if (guardrail.guardrail_name) {
                            toggleGuardrailSelection(guardrail.guardrail_name);
                          }
                        }}
                        className={cn(
                          "pl-6 pr-4 py-3 cursor-pointer hover:bg-muted transition-colors border-l-4",
                          isSelected
                            ? "bg-blue-50 dark:bg-blue-950/30 border-l-blue-500"
                            : "border-l-transparent",
                        )}
                      >
                        <div className="flex items-center space-x-2">
                          <FlaskConical className="h-4 w-4 text-muted-foreground" />
                          <span className="font-medium text-foreground">
                            {guardrail.guardrail_name}
                          </span>
                        </div>
                        <div className="text-xs space-y-1 mt-1">
                          <div>
                            <span className="font-medium">Type: </span>
                            <span className="text-muted-foreground">
                              {guardrail.litellm_params.guardrail}
                            </span>
                          </div>
                          <div>
                            <span className="font-medium">Mode: </span>
                            <span className="text-muted-foreground">
                              {guardrail.litellm_params.mode}
                            </span>
                          </div>
                        </div>
                      </li>
                    );
                  })}
                </ul>
              )}
            </div>

            <div className="p-3 border-t border-border bg-muted">
              <span className="text-xs text-muted-foreground">
                {selectedGuardrails.size} of {filteredGuardrails.length}{" "}
                selected
              </span>
            </div>
          </div>

          <div className="w-3/4 flex flex-col bg-background">
            <div className="p-4 border-b border-border flex justify-between items-center">
              <h2 className="text-xl font-semibold mb-0">
                Guardrail Testing Playground
              </h2>
            </div>

            <div className="flex-1 overflow-auto p-4">
              {selectedGuardrails.size === 0 ? (
                <div className="h-full flex flex-col items-center justify-center text-muted-foreground">
                  <FlaskConical className="h-12 w-12 mb-4" />
                  <p className="text-lg font-medium text-foreground mb-2">
                    Select Guardrails to Test
                  </p>
                  <p className="text-center text-muted-foreground max-w-md">
                    Choose one or more guardrails from the left sidebar to
                    start testing and comparing results.
                  </p>
                </div>
              ) : (
                <div className="h-full">
                  <GuardrailTestPanel
                    guardrailNames={Array.from(selectedGuardrails)}
                    onSubmit={handleTestGuardrails}
                    results={testResults.length > 0 ? testResults : null}
                    errors={testErrors.length > 0 ? testErrors : null}
                    isLoading={isTesting}
                    onClose={() => setSelectedGuardrails(new Set())}
                  />
                </div>
              )}
            </div>
          </div>
        </div>
      </Card>
    </div>
  );
};

export default GuardrailTestPlayground;
