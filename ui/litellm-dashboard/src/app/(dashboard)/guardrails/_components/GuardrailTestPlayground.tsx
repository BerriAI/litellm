import React, { useState } from "react";
import { FlaskConical, Search } from "lucide-react";
import GuardrailTestPanel from "./GuardrailTestPanel";
import { applyGuardrail } from "@/components/networking";
import { toast } from "@/lib/toast";
import { Card, CardContent } from "@/components/ui/card";
import { InputGroup, InputGroupAddon, InputGroupInput } from "@/components/ui/input-group";
import { UiLoadingSpinner } from "@/components/ui/ui-loading-spinner";
import { GuardrailMode } from "@/components/guardrails/types";
import { formatGuardrailMode } from "./guardrail_info_helpers";

interface GuardrailItem {
  guardrail_id?: string;
  guardrail_name: string | null;
  litellm_params: {
    guardrail: string;
    mode: GuardrailMode;
    default_on: boolean;
  };
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
  onClose,
}) => {
  const [selectedGuardrails, setSelectedGuardrails] = useState<Set<string>>(new Set());
  const [searchQuery, setSearchQuery] = useState("");
  const [testResults, setTestResults] = useState<TestResult[]>([]);
  const [testErrors, setTestErrors] = useState<TestError[]>([]);
  const [isTesting, setIsTesting] = useState(false);

  const filteredGuardrails = guardrailsList.filter((guardrail) =>
    guardrail.guardrail_name?.toLowerCase().includes(searchQuery.toLowerCase()),
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

  const handleTestGuardrails = async (text: string, metadata?: Record<string, unknown> | null) => {
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
          const result = await applyGuardrail(accessToken, guardrailName, text, null, null, metadata);
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
      toast.success(`${results.length} guardrail${results.length > 1 ? "s" : ""} applied successfully`);
    }
    if (errors.length > 0) {
      toast.fromError(`${errors.length} guardrail${errors.length > 1 ? "s" : ""} failed`);
    }
  };

  return (
    <div className="w-full h-[calc(100vh-200px)]">
      <Card className="h-full overflow-hidden py-0">
        <CardContent className="h-full p-0">
          <div className="flex h-full">
            {/* Left Sidebar - Guardrails List */}
            <div className="flex w-1/4 flex-col overflow-hidden border-r border-border">
              <div className="border-b border-border p-4">
                <div className="mb-3">
                  <h3 className="mb-3 text-lg font-semibold">Guardrails</h3>
                  <InputGroup>
                    <InputGroupAddon>
                      <Search className="size-4 text-muted-foreground" />
                    </InputGroupAddon>
                    <InputGroupInput
                      placeholder="Search guardrails..."
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                    />
                  </InputGroup>
                </div>
              </div>

              <div className="flex-1 overflow-auto">
                {isLoading ? (
                  <div className="flex h-32 items-center justify-center" aria-busy="true">
                    <UiLoadingSpinner className="size-6 text-muted-foreground" />
                  </div>
                ) : filteredGuardrails.length === 0 ? (
                  <div className="p-4 text-center text-muted-foreground">
                    {searchQuery ? "No guardrails match your search" : "No guardrails available"}
                  </div>
                ) : (
                  <ul className="m-0 list-none p-0">
                    {filteredGuardrails.map((guardrail) => (
                      <li
                        key={guardrail.guardrail_id ?? guardrail.guardrail_name}
                        onClick={() => {
                          if (guardrail.guardrail_name) {
                            toggleGuardrailSelection(guardrail.guardrail_name);
                          }
                        }}
                        className={`cursor-pointer border-b border-border py-3 pr-4 pl-6 transition-colors hover:bg-muted/40 ${
                          selectedGuardrails.has(guardrail.guardrail_name || "")
                            ? "border-l-4 border-l-primary bg-accent"
                            : "border-l-4 border-l-transparent"
                        }`}
                      >
                        <div className="flex items-center space-x-2">
                          <FlaskConical className="size-4 text-muted-foreground" />
                          <span className="font-medium">{guardrail.guardrail_name}</span>
                        </div>
                        <div className="mt-1 space-y-1 text-xs">
                          <div>
                            <span className="font-medium">Type: </span>
                            <span className="text-muted-foreground">{guardrail.litellm_params.guardrail}</span>
                          </div>
                          <div>
                            <span className="font-medium">Mode: </span>
                            <span className="text-muted-foreground">
                              {formatGuardrailMode(guardrail.litellm_params.mode)}
                            </span>
                          </div>
                        </div>
                      </li>
                    ))}
                  </ul>
                )}
              </div>

              <div className="border-t border-border bg-muted/40 p-3">
                <span className="text-xs text-muted-foreground">
                  {selectedGuardrails.size} of {filteredGuardrails.length} selected
                </span>
              </div>
            </div>

            {/* Right Panel - Test Area */}
            <div className="flex w-3/4 flex-col">
              <div className="flex items-center justify-between border-b border-border p-4">
                <h2 className="mb-0 text-xl font-semibold">Guardrail Testing Playground</h2>
              </div>

              <div className="flex-1 overflow-auto p-4">
                {selectedGuardrails.size === 0 ? (
                  <div className="flex h-full flex-col items-center justify-center text-muted-foreground">
                    <FlaskConical className="mb-4 size-12" />
                    <p className="mb-2 text-lg font-medium">Select Guardrails to Test</p>
                    <p className="max-w-md text-center">
                      Choose one or more guardrails from the left sidebar to start testing and comparing results.
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
        </CardContent>
      </Card>
    </div>
  );
};

export default GuardrailTestPlayground;
