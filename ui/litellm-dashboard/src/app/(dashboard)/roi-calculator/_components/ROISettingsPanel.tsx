"use client";

import React from "react";

import { apiClient } from "@/components/networking";
import { extractErrorMessage } from "@/utils/errorUtils";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import type { ROIRepository, ROIRepositoriesResponse, ROISettings, ROISettingsUpdate } from "./roiCalculatorData";

export default function ROISettingsPanel({
  accessToken,
  initialSettings,
  onboarding,
  onSaved,
  onStartSync,
  readOnly,
  syncDisabled,
}: {
  accessToken: string | null;
  initialSettings: ROISettings;
  onboarding: boolean;
  onSaved: (settings: ROISettings) => void;
  onStartSync: () => Promise<void>;
  readOnly: boolean;
  syncDisabled: boolean;
}) {
  const [apiUrl, setApiUrl] = React.useState(initialSettings.github_api_url);
  const [token, setToken] = React.useState("");
  const [clearToken, setClearToken] = React.useState(false);
  const [repos, setRepos] = React.useState(initialSettings.repos);
  const [model, setModel] = React.useState(initialSettings.estimator_model);
  const [prompt, setPrompt] = React.useState(initialSettings.estimator_prompt);
  const [backfillDays, setBackfillDays] = React.useState(String(initialSettings.backfill_days));
  const [repositoryQuery, setRepositoryQuery] = React.useState("");
  const [repositoryPage, setRepositoryPage] = React.useState(1);
  const [availableRepos, setAvailableRepos] = React.useState<ROIRepository[]>([]);
  const [hasMoreRepos, setHasMoreRepos] = React.useState(false);
  const [busy, setBusy] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);
  const [message, setMessage] = React.useState<string | null>(null);

  const canLoadRepositories =
    initialSettings.has_github_token && !token.trim() && apiUrl === initialSettings.github_api_url;

  const loadRepositories = async (page: number) => {
    if (!accessToken || !canLoadRepositories) return;
    try {
      setBusy(true);
      const response: ROIRepositoriesResponse = await apiClient.get("/roi-calculator/repositories", {
        accessToken,
        query: { query: repositoryQuery, page },
      });
      setAvailableRepos((current) => (page === 1 ? response.repositories : [...current, ...response.repositories]));
      setHasMoreRepos(response.has_more);
      setRepositoryPage(page);
      setError(null);
    } catch (reason) {
      setError(extractErrorMessage(reason));
    } finally {
      setBusy(false);
    }
  };

  const saveSettings = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!accessToken || readOnly) return;
    const body: ROISettingsUpdate = {
      github_api_url: apiUrl,
      repos,
      estimator_model: model,
      estimator_prompt: prompt,
      backfill_days: Number(backfillDays),
      ...(clearToken ? { github_token: null } : {}),
      ...(token.trim() ? { github_token: token.trim() } : {}),
    };
    try {
      setBusy(true);
      const updated: ROISettings = await apiClient.put("/roi-calculator/settings", { accessToken, body });
      onSaved(updated);
      setToken("");
      setClearToken(false);
      setMessage("Settings saved.");
      setError(null);
    } catch (reason) {
      setError(extractErrorMessage(reason));
      setMessage(null);
    } finally {
      setBusy(false);
    }
  };

  const toggleRepository = (name: string) => {
    setRepos((current) => (current.includes(name) ? current.filter((repo) => repo !== name) : [...current, name]));
  };

  return (
    <Card>
      <CardHeader>
        <h2 className="text-base leading-normal font-medium">
          {onboarding ? "Connect GitHub to get started" : "ROI Calculator settings"}
        </h2>
        <CardDescription>
          {onboarding
            ? "Save a GitHub token, choose repositories and a router model, then run the analysis."
            : "Choose GitHub repositories and the router model used for metadata-only estimates."}
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-5">
        {error && (
          <p className="text-sm text-destructive" role="alert">
            {error}
          </p>
        )}
        {message && (
          <p className="text-sm text-emerald-700" role="status">
            {message}
          </p>
        )}
        <form className="space-y-5" onSubmit={(event) => void saveSettings(event)}>
          <div className="grid gap-2">
            <Label htmlFor="roi-github-url">GitHub API URL</Label>
            <Input
              disabled={readOnly}
              id="roi-github-url"
              value={apiUrl}
              onChange={(event) => setApiUrl(event.target.value)}
            />
          </div>
          <div className="grid gap-2">
            <Label htmlFor="roi-github-token">GitHub token</Label>
            <Input
              autoComplete="new-password"
              disabled={readOnly}
              id="roi-github-token"
              type="password"
              value={token}
              onChange={(event) => {
                setToken(event.target.value);
                setClearToken(false);
              }}
              placeholder={initialSettings.has_github_token ? "Token saved" : "Enter a GitHub token"}
            />
            <p className="text-xs text-muted-foreground">
              {initialSettings.has_github_token
                ? "A token is saved securely and is never shown here."
                : "Save a token to list repositories and read private repository metadata."}
            </p>
            {initialSettings.has_github_token && apiUrl !== initialSettings.github_api_url && !token.trim() && (
              <p className="text-xs text-amber-700">
                Changing the GitHub API URL clears the saved token. Enter a replacement token to keep access.
              </p>
            )}
            {initialSettings.has_github_token && (
              <label className="flex items-center gap-2 text-sm">
                <input
                  aria-label="Clear saved GitHub token"
                  checked={clearToken}
                  disabled={readOnly}
                  type="checkbox"
                  onChange={(event) => setClearToken(event.target.checked)}
                />
                Clear saved token
              </label>
            )}
          </div>
          <div className="grid gap-2">
            <Label htmlFor="roi-repository-search">Repositories</Label>
            <div className="flex gap-2">
              <Input
                id="roi-repository-search"
                value={repositoryQuery}
                onChange={(event) => setRepositoryQuery(event.target.value)}
                placeholder="Search repositories"
              />
              <Button
                type="button"
                variant="outline"
                disabled={busy || !canLoadRepositories}
                onClick={() => void loadRepositories(1)}
              >
                Load repositories
              </Button>
            </div>
            {!canLoadRepositories && (
              <p className="text-xs text-muted-foreground">
                Save the GitHub token and API URL before loading repositories.
              </p>
            )}
            {repos.length > 0 && <p className="text-sm text-muted-foreground">Selected: {repos.join(", ")}</p>}
            <div className="max-h-56 space-y-2 overflow-y-auto rounded-md border p-3">
              {availableRepos.map((repository) => (
                <label key={repository.name} className="flex items-center gap-2 text-sm">
                  <input
                    aria-label={`Select ${repository.name}`}
                    checked={repos.includes(repository.name)}
                    disabled={readOnly}
                    type="checkbox"
                    onChange={() => toggleRepository(repository.name)}
                  />
                  <span>{repository.name}</span>
                  <span className="text-xs text-muted-foreground">
                    {repository.visibility}
                    {repository.archived ? " · archived" : ""}
                  </span>
                </label>
              ))}
              {availableRepos.length === 0 && (
                <p className="text-sm text-muted-foreground">
                  Load repositories to choose which pull requests to analyze.
                </p>
              )}
            </div>
            {hasMoreRepos && (
              <Button
                type="button"
                variant="link"
                className="w-fit px-0"
                disabled={busy}
                onClick={() => void loadRepositories(repositoryPage + 1)}
              >
                Load more repositories
              </Button>
            )}
          </div>
          <div className="grid gap-2">
            <Label htmlFor="roi-estimator-model">Estimator model</Label>
            <select
              id="roi-estimator-model"
              className="h-9 rounded-md border bg-background px-3 text-sm"
              disabled={readOnly}
              value={model}
              onChange={(event) => setModel(event.target.value)}
            >
              <option value="">Select a router model</option>
              {model && !initialSettings.available_models.includes(model) && <option value={model}>{model}</option>}
              {initialSettings.available_models.map((availableModel) => (
                <option key={availableModel} value={availableModel}>
                  {availableModel}
                </option>
              ))}
            </select>
          </div>
          <div className="grid gap-2">
            <Label htmlFor="roi-estimator-prompt">Estimator prompt</Label>
            <Textarea
              id="roi-estimator-prompt"
              rows={5}
              disabled={readOnly}
              value={prompt}
              onChange={(event) => setPrompt(event.target.value)}
            />
          </div>
          <div className="grid max-w-xs gap-2">
            <Label htmlFor="roi-backfill-days">Backfill days</Label>
            <Input
              id="roi-backfill-days"
              min={1}
              max={3650}
              type="number"
              disabled={readOnly}
              value={backfillDays}
              onChange={(event) => setBackfillDays(event.target.value)}
            />
          </div>
          {!readOnly && (
            <div className="flex flex-wrap gap-2">
              <Button disabled={busy} type="submit">
                {busy ? "Saving…" : "Save settings"}
              </Button>
              <Button
                disabled={!initialSettings.ready || syncDisabled || busy}
                type="button"
                variant="outline"
                onClick={() => void onStartSync()}
              >
                Run analysis
              </Button>
            </div>
          )}
        </form>
      </CardContent>
    </Card>
  );
}
