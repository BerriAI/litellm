"use client";

import React from "react";

import { SearchSelect } from "@/components/shared/SearchSelect";
import { estimatorModelOptions } from "./roiCalculatorData";

import { apiClient } from "@/components/networking";
import { extractErrorMessage } from "@/utils/errorUtils";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "@/components/ui/dialog";
import { Textarea } from "@/components/ui/textarea";
import type { ROIRepository, ROIRepositoriesResponse, ROISettings, ROISettingsUpdate } from "./roiCalculatorData";

export default function ROISettingsPanel({
  accessToken,
  initialSettings,
  onboarding,
  onSaved,
  onReset,
  onStartSync,
  readOnly,
  syncDisabled,
}: {
  accessToken: string | null;
  initialSettings: ROISettings;
  onboarding: boolean;
  onSaved: (settings: ROISettings) => void;
  onReset: (settings: ROISettings) => void;
  onStartSync: () => Promise<void>;
  readOnly: boolean;
  syncDisabled: boolean;
}) {
  const [provider, setProvider] = React.useState<"github" | "gitlab">(initialSettings.source_provider ?? "github");
  const sourceName = provider === "gitlab" ? "GitLab" : "GitHub";
  const savedToken = provider === "gitlab" ? initialSettings.has_gitlab_token : initialSettings.has_github_token;
  const savedUrl = provider === "gitlab" ? initialSettings.gitlab_api_url : initialSettings.github_api_url;
  const initialStep = savedToken ? 1 : 0;
  const [step, setStep] = React.useState(initialSettings.ready ? 2 : initialStep);
  const [apiUrl, setApiUrl] = React.useState(savedUrl ?? "https://gitlab.com/api/v4");
  const [token, setToken] = React.useState("");
  const [clearToken, setClearToken] = React.useState(false);
  const [repos, setRepos] = React.useState(initialSettings.repos);
  const [model, setModel] = React.useState(initialSettings.estimator_model);
  const [prompt, setPrompt] = React.useState(initialSettings.estimator_prompt);
  const [backfillDays, setBackfillDays] = React.useState(String(initialSettings.backfill_days));
  const [intervalHours, setIntervalHours] = React.useState(
    String((initialSettings.update_interval_minutes ?? 1440) / 60),
  );
  const [estimatorKey, setEstimatorKey] = React.useState("");
  const [clearEstimatorKey, setClearEstimatorKey] = React.useState(false);
  const [repositoryName, setRepositoryName] = React.useState("");
  const [resetOpen, setResetOpen] = React.useState(false);
  const [repositoryQuery, setRepositoryQuery] = React.useState("");
  const [repositoryPage, setRepositoryPage] = React.useState(1);
  const [availableRepos, setAvailableRepos] = React.useState<ROIRepository[]>([]);
  const [hasMoreRepos, setHasMoreRepos] = React.useState(false);
  const [busy, setBusy] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);
  const [message, setMessage] = React.useState<string | null>(null);

  const sourceUnchanged = provider === (initialSettings.source_provider ?? "github") && apiUrl === savedUrl;
  const credentialsSaved = sourceUnchanged && !token.trim() && !clearToken;
  const canLoadRepositories = credentialsSaved && (provider === "gitlab" || savedToken);
  const tokenHelp =
    provider === "gitlab"
      ? "For private projects, use a token with read_api scope and project access."
      : "For private repositories, use a token with read access to contents and pull requests.";

  const changeProvider = (next: "github" | "gitlab") => {
    setProvider(next);
    setApiUrl(
      next === "gitlab"
        ? initialSettings.gitlab_api_url ?? "https://gitlab.com/api/v4"
        : initialSettings.github_api_url,
    );
    setToken("");
    setClearToken(false);
    setRepos([]);
    setAvailableRepos([]);
    setHasMoreRepos(false);
  };

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

  const saveSettings = async () => {
    if (!accessToken || readOnly) return false;
    const tokenValue = clearToken ? null : token.trim() || undefined;
    const body: ROISettingsUpdate = {
      source_provider: provider,
      ...(provider === "gitlab" ? { gitlab_api_url: apiUrl } : { github_api_url: apiUrl }),
      repos,
      estimator_model: model,
      estimator_prompt: prompt,
      backfill_days: Number(backfillDays),
      update_interval_minutes: Number(intervalHours) * 60,
      ...(clearEstimatorKey ? { estimator_key: null } : {}),
      ...(estimatorKey.trim() ? { estimator_key: estimatorKey.trim() } : {}),
      ...(provider === "gitlab" ? { gitlab_token: tokenValue } : { github_token: tokenValue }),
    };
    try {
      setBusy(true);
      const updated: ROISettings = await apiClient.put("/roi-calculator/settings", { accessToken, body });
      onSaved(updated);
      setToken("");
      setEstimatorKey("");
      setClearEstimatorKey(false);
      setClearToken(false);
      setMessage("Settings saved.");
      setError(null);
      return true;
    } catch (reason) {
      setError(extractErrorMessage(reason));
      setMessage(null);
      return false;
    } finally {
      setBusy(false);
    }
  };

  const submit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!(await saveSettings())) return;
    if (onboarding && step === 0) {
      setStep(1);
      if (!(token.trim() || savedToken) || clearToken) return;
      try {
        const result = await apiClient.get<ROIRepositoriesResponse>("/roi-calculator/repositories", { accessToken });
        setAvailableRepos(result.repositories);
        setHasMoreRepos(result.has_more);
        setRepositoryPage(1);
        setStep(1);
      } catch (reason) {
        setError(extractErrorMessage(reason));
      }
    } else if (onboarding && step === 1) setStep(2);
    else if (onboarding) await onStartSync();
  };

  const saveAndRun = async () => {
    if (await saveSettings()) await onStartSync();
  };

  const testConnections = async () => {
    if (!(await saveSettings())) return;
    setBusy(true);
    try {
      await apiClient.post("/roi-calculator/connections/test", { accessToken });
      setMessage("Gateway model and selected repositories are available.");
    } catch (reason) {
      setError(extractErrorMessage(reason));
    } finally {
      setBusy(false);
    }
  };

  const resetSetup = async () => {
    setBusy(true);
    try {
      const updated = await apiClient.post<ROISettings>("/roi-calculator/setup/reset", { accessToken });
      setRepos([]);
      setStep(0);
      setResetOpen(false);
      onReset(updated);
    } catch (reason) {
      setError(extractErrorMessage(reason));
    } finally {
      setBusy(false);
    }
  };

  const toggleRepository = (name: string) => {
    setRepos((current) => (current.includes(name) ? current.filter((repo) => repo !== name) : [...current, name]));
  };

  const formDisabled = busy || syncDisabled;
  const runDisabled = formDisabled || !repos.length || !model;
  const sourceUrlChanged = apiUrl !== savedUrl;
  const missingReplacementToken = savedToken && sourceUrlChanged && !token.trim();
  const stepReady = [true, repos.length > 0, Boolean(model)][step];
  const onboardingLabel = step < 2 ? "Continue" : "Start backfill";
  const submitLabel = onboarding ? onboardingLabel : "Save settings";

  return (
    <Card className={onboarding ? "max-w-2xl" : "border-0 py-0 shadow-none ring-0"}>
      {onboarding && (
        <CardHeader>
          <h2 className="text-base leading-normal font-medium">
            {["Connect your repositories", "Choose repositories", "Choose an estimator"][step]}
          </h2>
          <CardDescription>
            Your gateway is already connected. Choose a source and an estimator for your first report.
          </CardDescription>
        </CardHeader>
      )}
      <CardContent className={onboarding ? "space-y-5" : "space-y-5 px-0"}>
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
        {onboarding && (
          <p className="text-sm text-muted-foreground">Step {step + 1} of 3 · Source / Repositories / Estimator</p>
        )}
        <form className="space-y-5" onSubmit={(event) => void submit(event)}>
          <fieldset disabled={busy || syncDisabled || readOnly} className="space-y-5">
            {(!onboarding || step === 0) && (
              <section className="space-y-4">
                {!onboarding && <h3 className="font-semibold">Connection</h3>}
                <div className="grid gap-2">
                  <Label htmlFor="roi-source">Repository source</Label>
                  <select
                    id="roi-source"
                    className="h-9 min-w-0 rounded-md border bg-background pl-3 pr-9 text-sm"
                    value={provider}
                    onChange={(event) => changeProvider(event.target.value as "github" | "gitlab")}
                  >
                    <option value="github">GitHub</option>
                    <option value="gitlab">GitLab</option>
                  </select>
                  {provider !== (initialSettings.source_provider ?? "github") && !onboarding && (
                    <p className="text-xs text-muted-foreground">
                      Switching source starts a new report and resets email matches.
                    </p>
                  )}
                </div>
                <details>
                  <summary className="cursor-pointer text-sm text-muted-foreground">Self-hosted {sourceName}</summary>
                  <div className="mt-3 grid gap-2">
                    <Label htmlFor="roi-github-url">{sourceName} API URL</Label>
                    <Input
                      disabled={readOnly}
                      id="roi-github-url"
                      value={apiUrl}
                      onChange={(event) => setApiUrl(event.target.value)}
                    />
                  </div>
                </details>
                <div className="grid gap-2">
                  <Label htmlFor="roi-github-token">{sourceName} token (optional for public repositories)</Label>
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
                    placeholder={savedToken ? "Token saved" : `Enter a ${sourceName} token`}
                  />
                  <p className="text-xs text-muted-foreground">
                    {savedToken ? "A token is saved securely and is never shown here." : tokenHelp}
                  </p>
                  {missingReplacementToken && (
                    <p className="text-xs text-amber-700">
                      Changing the API URL clears the saved token. Enter a replacement token to keep access.
                    </p>
                  )}
                  {savedToken && (
                    <label className="flex items-center gap-2 text-sm">
                      <input
                        aria-label={`Clear saved ${sourceName} token`}
                        checked={clearToken}
                        disabled={readOnly}
                        type="checkbox"
                        onChange={(event) => setClearToken(event.target.checked)}
                      />
                      Clear saved token
                    </label>
                  )}
                </div>
              </section>
            )}
            {(!onboarding || step === 1) && (
              <section className={onboarding ? "space-y-4" : "space-y-4 border-t pt-5"}>
                <h3 className="font-semibold">Repositories</h3>
                <Label className="sr-only" htmlFor="roi-repository-search">
                  Search repositories
                </Label>
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
                    {provider === "github" && !savedToken
                      ? "Save a GitHub token to browse repositories, or add a public repository by name."
                      : "Save the source and connection settings before loading repositories."}
                  </p>
                )}
                {repos.length > 0 && (
                  <div className="flex flex-wrap gap-2">
                    {repos.map((repo) => (
                      <Button
                        key={repo}
                        type="button"
                        variant="outline"
                        disabled={readOnly}
                        onClick={() => toggleRepository(repo)}
                        aria-label={`Remove ${repo}`}
                      >
                        {repo} ×
                      </Button>
                    ))}
                  </div>
                )}
                <div>
                  <Label htmlFor="roi-repository-name">Add a repository by name</Label>
                  <div className="mt-2 flex gap-2">
                    <Input
                      id="roi-repository-name"
                      aria-label="Repository name"
                      placeholder={provider === "gitlab" ? "group/subgroup/project" : "owner/repository"}
                      value={repositoryName}
                      onChange={(e) => setRepositoryName(e.target.value)}
                    />
                    <Button
                      type="button"
                      variant="outline"
                      disabled={!repositoryName.trim()}
                      onClick={() => {
                        if (!repos.includes(repositoryName.trim())) setRepos([...repos, repositoryName.trim()]);
                        setRepositoryName("");
                      }}
                    >
                      Add
                    </Button>
                  </div>
                </div>
                {availableRepos.length > 0 && (
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
                  </div>
                )}
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
              </section>
            )}
            {(!onboarding || step === 2) && (
              <section className={onboarding ? "space-y-4" : "space-y-4 border-t pt-5"}>
                {!onboarding && <h3 className="font-semibold">Estimation and updates</h3>}
                <div className="grid gap-2">
                  <Label htmlFor="roi-estimator-model">Estimator model</Label>
                  <SearchSelect
                    inputId="roi-estimator-model"
                    options={estimatorModelOptions(initialSettings)}
                    value={model}
                    onValueChange={(value) => setModel(value ?? "")}
                    placeholder="Search estimator models"
                    emptyText="No matching models configured on this gateway"
                    disabled={readOnly}
                    className="h-9"
                  />
                  <p className="text-xs text-muted-foreground">
                    We recommend GPT-6 Luna for estimating PR effort. Choose a model configured on your gateway.
                  </p>
                </div>
                <details>
                  <summary className="cursor-pointer text-sm text-muted-foreground">Advanced estimator options</summary>
                  <div className="mt-3 grid gap-2">
                    <Label htmlFor="roi-estimator-prompt">Estimator prompt</Label>
                    <Textarea
                      id="roi-estimator-prompt"
                      rows={5}
                      disabled={readOnly}
                      value={prompt}
                      onChange={(event) => setPrompt(event.target.value)}
                    />
                  </div>
                </details>
                <div className="grid gap-4 sm:grid-cols-2">
                  <div className="grid content-start gap-2">
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
                  <div className="grid gap-2">
                    <Label htmlFor="roi-interval">Update interval (hours)</Label>
                    <Input
                      id="roi-interval"
                      type="number"
                      min={0}
                      max={720}
                      step="any"
                      required
                      value={intervalHours}
                      onChange={(e) => setIntervalHours(e.target.value)}
                    />
                    <p className="text-xs text-muted-foreground">
                      Use 0 for manual updates. Automatic updates require at least 5 minutes and a running gateway.
                    </p>
                  </div>
                </div>
                <details>
                  <summary className="cursor-pointer text-sm text-muted-foreground">Advanced settings</summary>
                  <div className="mt-3 space-y-3">
                    <Label htmlFor="roi-estimator-key">Estimator API key</Label>
                    <Input
                      id="roi-estimator-key"
                      type="password"
                      autoComplete="new-password"
                      value={estimatorKey}
                      onChange={(e) => {
                        setEstimatorKey(e.target.value);
                        setClearEstimatorKey(false);
                      }}
                      placeholder={initialSettings.has_estimator_key ? "Key saved" : "Optional gateway key"}
                    />
                    <p className="text-xs text-muted-foreground">
                      Defaults to the gateway admin key. Use a dedicated inference key to separate estimation costs from
                      people&apos;s spend.
                    </p>
                    {initialSettings.has_estimator_key && (
                      <label className="flex items-center gap-2 text-sm">
                        <input
                          type="checkbox"
                          checked={clearEstimatorKey}
                          onChange={(e) => setClearEstimatorKey(e.target.checked)}
                        />
                        Use gateway admin key instead
                      </label>
                    )}
                    <Button type="button" variant="link" onClick={() => setPrompt(initialSettings.default_prompt)}>
                      Reset prompt
                    </Button>
                    {!onboarding && !readOnly && (
                      <Button type="button" variant="outline" onClick={() => setResetOpen(true)}>
                        Restart setup
                      </Button>
                    )}
                  </div>
                </details>
                <p className="text-xs text-muted-foreground">
                  Estimates use descriptions, file counts, and commit messages, without source code. Hours represent
                  estimated effort without AI, not measured hours saved.
                </p>
              </section>
            )}
            {!readOnly && (
              <div className="flex flex-wrap gap-2 border-t pt-5">
                {onboarding && step > 0 && (
                  <Button
                    type="button"
                    variant="outline"
                    disabled={busy || syncDisabled}
                    onClick={() => setStep(step - 1)}
                  >
                    Back
                  </Button>
                )}
                <Button disabled={formDisabled || (onboarding && !stepReady)} type="submit">
                  {busy ? "Saving…" : submitLabel}
                </Button>
                {!onboarding && (
                  <Button type="button" variant="outline" onClick={() => void testConnections()}>
                    Test connections
                  </Button>
                )}
                {!onboarding && (
                  <Button disabled={runDisabled} type="button" variant="outline" onClick={() => void saveAndRun()}>
                    Save and run analysis
                  </Button>
                )}
              </div>
            )}
          </fieldset>
        </form>
        <Dialog open={resetOpen} onOpenChange={setResetOpen}>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Restart setup?</DialogTitle>
              <DialogDescription>
                Clear reports and repository selections. Saved connections and cached estimates will be kept.
              </DialogDescription>
            </DialogHeader>
            <DialogFooter>
              <Button variant="outline" disabled={busy} onClick={() => setResetOpen(false)}>
                Cancel
              </Button>
              <Button disabled={busy} onClick={() => void resetSetup()}>
                Restart setup
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </CardContent>
    </Card>
  );
}
