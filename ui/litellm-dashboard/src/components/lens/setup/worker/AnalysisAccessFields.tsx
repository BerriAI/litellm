import type { AnalysisAccess } from "./workerSchema";
("use client");

import { lensQueries } from "../../api/queries";

import { useQuery } from "@tanstack/react-query";
import { useLensApi } from "../../api/useLensApi";
import { SearchSelect } from "@/components/shared/SearchSelect";
import { Input } from "@/components/ui/input";

export function AnalysisAccessFields({
  accessToken,
  value,
  onChange,
}: {
  accessToken: string;
  value: AnalysisAccess;
  onChange: (value: AnalysisAccess) => void;
}) {
  const apiClient = useLensApi();
  const models = useQuery(lensQueries.models(apiClient, accessToken));
  return (
    <div className="space-y-5">
      <div className="space-y-2">
        <label htmlFor="analysis-access-model" className="block text-sm font-medium">
          Analysis model
        </label>
        <SearchSelect
          inputId="analysis-access-model"
          options={(models.data?.data ?? []).map(({ id }) => ({ label: id, value: id }))}
          value={value.model}
          onValueChange={(model) => onChange({ ...value, model })}
          placeholder={models.isLoading ? "Loading models…" : "Select a model"}
        />
      </div>
      <div className="space-y-2">
        <label htmlFor="analysis-access-budget" className="block text-sm font-medium">
          Monthly limit (USD)
        </label>
        <Input
          id="analysis-access-budget"
          type="number"
          min="0.01"
          step="0.01"
          value={value.budget}
          onChange={(e) => onChange({ ...value, budget: e.target.value })}
        />
        <p className="text-xs text-muted-foreground">Shared across all investigations.</p>
      </div>
      {models.error && (
        <p role="alert" className="text-sm text-destructive">
          {models.error.message}
        </p>
      )}
    </div>
  );
}
