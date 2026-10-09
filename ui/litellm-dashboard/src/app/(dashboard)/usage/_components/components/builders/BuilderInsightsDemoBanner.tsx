"use client";

import { Check, Info } from "lucide-react";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/cva.config";
import type { BuilderInsightBuilder } from "./builderInsightsData";

const STEPS = ["Connect GitHub", "Pick repos", "Map users", "Review"] as const;
const STEP_DESCRIPTIONS = [
  "Choose a connection method and enter your GitHub organization.",
  "Choose which repositories contribute merged PRs.",
  "Confirm the GitHub login for each LiteLLM user.",
  "Review the setup. Sync is not available yet.",
] as const;
const SAMPLE_REPOSITORIES = ["api", "web", "infra", "sdk"] as const;
const TOKEN_URL =
  "https://github.com/settings/tokens/new?scopes=repo,read:org&description=LiteLLM%20Builder%20Insights";

type SetupStep = 0 | 1 | 2 | 3;
type ConnectionMethod = "app" | "token";
type BuilderEmail = Pick<BuilderInsightBuilder, "email">;

const githubLoginForEmail = (email: string): string => email.split("@", 1)[0] ?? "";

export function BuilderInsightsDemoBanner({ builders }: { builders: readonly BuilderEmail[] }) {
  const [open, setOpen] = useState(false);
  const [step, setStep] = useState<SetupStep>(0);
  const [method, setMethod] = useState<ConnectionMethod>("app");
  const [organization, setOrganization] = useState("");
  const [token, setToken] = useState("");
  const [selectedRepos, setSelectedRepos] = useState<readonly string[]>([]);
  const [mappingOverrides, setMappingOverrides] = useState<Readonly<Record<string, string>>>({});
  const [savedOrganization, setSavedOrganization] = useState<string | null>(null);

  const trimmedOrganization = organization.trim();
  const mappedUsers = builders.map(({ email }) => ({
    email,
    login: mappingOverrides[email] ?? githubLoginForEmail(email),
  }));
  const matchedUsers = mappedUsers.filter(({ login }) => login.trim().length > 0).length;
  const canContinue = trimmedOrganization.length > 0 && (method === "app" || token.trim().length > 0);
  const allReposSelected = selectedRepos.length === SAMPLE_REPOSITORIES.length;

  const handleOpenChange = (nextOpen: boolean) => {
    setOpen(nextOpen);
    if (!nextOpen) {
      setToken("");
      setStep(0);
    }
  };

  const finishSetup = () => {
    setSavedOrganization(trimmedOrganization);
    handleOpenChange(false);
  };

  const toggleRepository = (repository: string, checked: boolean) => {
    setSelectedRepos((current) => {
      if (!checked) return current.filter((item) => item !== repository);
      if (current.includes(repository)) return current;
      return [...current, repository];
    });
  };

  return (
    <>
      <div className="flex flex-wrap items-center justify-between gap-3 rounded-xl border bg-card px-4 py-3">
        <div className="flex min-w-0 items-center gap-2.5 text-sm text-muted-foreground">
          <Info aria-hidden="true" className="size-4 shrink-0" />
          <p role="status">
            {savedOrganization
              ? `GitHub setup saved for ${savedOrganization}. Sample data shows until sync ships.`
              : "Demo data. Builder Insights joins SpendLogs with merged PRs from GitHub to show your team."}
          </p>
        </div>
        <Button type="button" variant="outline" size="sm" onClick={() => setOpen(true)}>
          How to connect
        </Button>
      </div>
      <Dialog open={open} onOpenChange={handleOpenChange}>
        <DialogContent className="max-h-[calc(100dvh-2rem)] overflow-y-auto sm:max-w-2xl">
          <DialogHeader>
            <DialogTitle>{STEPS[step]}</DialogTitle>
            <DialogDescription>{STEP_DESCRIPTIONS[step]}</DialogDescription>
          </DialogHeader>

          <ol aria-label="Setup progress" className="grid grid-cols-2 gap-2 sm:grid-cols-4">
            {STEPS.map((label, index) => {
              const isComplete = index < step;
              const isCurrent = index === step;

              return (
                <li
                  key={label}
                  aria-current={isCurrent ? "step" : undefined}
                  className={cn(
                    "flex min-w-0 items-center gap-2 rounded-md px-2 py-1.5 text-xs",
                    isCurrent ? "bg-muted font-medium text-foreground" : "text-muted-foreground",
                  )}
                >
                  <span
                    className={cn(
                      "flex size-5 shrink-0 items-center justify-center rounded-full text-[10px] tabular-nums",
                      isComplete ? "bg-primary text-primary-foreground" : "bg-muted text-muted-foreground",
                      isCurrent && "border border-primary text-primary",
                    )}
                  >
                    {isComplete ? <Check aria-hidden="true" className="size-3" /> : index + 1}
                  </span>
                  <span className="truncate">{label}</span>
                </li>
              );
            })}
          </ol>

          {step === 0 && (
            <div className="grid gap-5">
              <div className="grid gap-3 sm:grid-cols-2">
                <button
                  type="button"
                  aria-pressed={method === "app"}
                  onClick={() => setMethod("app")}
                  className={cn(
                    "grid gap-2 rounded-lg border p-4 text-left transition-colors",
                    method === "app" ? "border-primary bg-muted/50" : "hover:bg-muted/50",
                  )}
                >
                  <span className="font-medium">GitHub App</span>
                  <span className="text-sm text-muted-foreground">
                    Recommended. Read-only access to PRs and commits.
                  </span>
                  <span className="text-xs text-muted-foreground">Available when sync ships</span>
                </button>
                <button
                  type="button"
                  aria-pressed={method === "token"}
                  onClick={() => setMethod("token")}
                  className={cn(
                    "grid gap-2 rounded-lg border p-4 text-left transition-colors",
                    method === "token" ? "border-primary bg-muted/50" : "hover:bg-muted/50",
                  )}
                >
                  <span className="font-medium">Personal access token</span>
                  <span className="text-sm text-muted-foreground">
                    Use one admin token for read-only repository access.
                  </span>
                </button>
              </div>

              {method === "app" ? (
                <div className="flex flex-wrap items-center gap-3">
                  <Button type="button" variant="outline" disabled>
                    Install GitHub App
                  </Button>
                  <span className="text-xs text-muted-foreground">Available when sync ships</span>
                </div>
              ) : (
                <div className="grid gap-2">
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <label htmlFor="builder-insights-token" className="text-sm font-medium">
                      Personal access token
                    </label>
                    <Button
                      type="button"
                      nativeButton={false}
                      variant="link"
                      size="sm"
                      render={<a href={TOKEN_URL} target="_blank" rel="noopener noreferrer" />}
                    >
                      Create token on GitHub
                    </Button>
                  </div>
                  <Input
                    id="builder-insights-token"
                    aria-label="Personal access token"
                    type="password"
                    autoComplete="off"
                    value={token}
                    onChange={(event) => setToken(event.target.value)}
                  />
                  <p className="text-xs text-muted-foreground">
                    This token stays in this dialog and is cleared when you close it.
                  </p>
                </div>
              )}

              <label htmlFor="builder-insights-organization" className="grid gap-2 text-sm font-medium">
                GitHub organization
                <Input
                  id="builder-insights-organization"
                  aria-label="GitHub organization"
                  placeholder="your-org"
                  value={organization}
                  onChange={(event) => setOrganization(event.target.value)}
                />
              </label>
            </div>
          )}

          {step === 1 && (
            <div className="grid gap-4">
              <div className="flex items-center gap-2">
                <Checkbox
                  id="builder-insights-select-all"
                  aria-label="Select all"
                  checked={allReposSelected}
                  onCheckedChange={(checked) => setSelectedRepos(checked === true ? [...SAMPLE_REPOSITORIES] : [])}
                />
                <label htmlFor="builder-insights-select-all" className="text-sm font-medium">
                  Select all
                </label>
              </div>
              <div className="grid gap-2 rounded-lg border p-3 sm:grid-cols-2">
                {SAMPLE_REPOSITORIES.map((repository) => (
                  <div key={repository} className="flex items-center gap-2">
                    <Checkbox
                      id={`builder-insights-repo-${repository}`}
                      aria-label={`${trimmedOrganization}/${repository}`}
                      checked={selectedRepos.includes(repository)}
                      onCheckedChange={(checked) => toggleRepository(repository, checked === true)}
                    />
                    <label htmlFor={`builder-insights-repo-${repository}`} className="text-sm text-muted-foreground">
                      {trimmedOrganization}/{repository}
                    </label>
                  </div>
                ))}
              </div>
              <p className="text-xs text-muted-foreground">Merged PRs from these repos count toward each builder</p>
            </div>
          )}

          {step === 2 && (
            <div className="grid gap-3">
              <div className="overflow-hidden rounded-lg border">
                <div className="grid grid-cols-[minmax(0,1fr)_minmax(0,0.8fr)] gap-3 bg-muted/40 px-3 py-2 text-xs font-medium text-muted-foreground">
                  <span>LiteLLM user email</span>
                  <span>GitHub login</span>
                </div>
                <div className="divide-y">
                  {mappedUsers.map(({ email, login }, index) => (
                    <div
                      key={email}
                      className="grid grid-cols-[minmax(0,1fr)_minmax(0,0.8fr)] items-center gap-3 px-3 py-2"
                    >
                      <span className="truncate text-sm">{email}</span>
                      <Input
                        id={`builder-insights-login-${index}`}
                        aria-label={`GitHub login for ${email}`}
                        autoComplete="off"
                        value={login}
                        onChange={(event) =>
                          setMappingOverrides((current) => ({ ...current, [email]: event.target.value }))
                        }
                      />
                    </div>
                  ))}
                </div>
              </div>
              <p className="text-xs text-muted-foreground">
                {matchedUsers} of {mappedUsers.length} matched
              </p>
            </div>
          )}

          {step === 3 && (
            <div className="grid gap-4">
              <dl className="grid gap-3 rounded-lg border p-4 sm:grid-cols-2">
                <div className="grid gap-1">
                  <dt className="text-xs text-muted-foreground">Connection method</dt>
                  <dd className="text-sm font-medium">{method === "app" ? "GitHub App" : "Personal access token"}</dd>
                </div>
                <div className="grid gap-1">
                  <dt className="text-xs text-muted-foreground">GitHub organization</dt>
                  <dd className="text-sm font-medium">{trimmedOrganization}</dd>
                </div>
                <div className="grid gap-1">
                  <dt className="text-xs text-muted-foreground">Repositories</dt>
                  <dd className="text-sm font-medium">
                    {selectedRepos.length} {selectedRepos.length === 1 ? "repo" : "repos"}
                  </dd>
                </div>
                <div className="grid gap-1">
                  <dt className="text-xs text-muted-foreground">Users mapped</dt>
                  <dd className="text-sm font-medium">
                    {matchedUsers} of {mappedUsers.length}
                  </dd>
                </div>
              </dl>
              <p className="rounded-lg bg-muted/50 p-3 text-sm leading-6 text-muted-foreground">
                Sync ships in the next release. Until then this page shows sample data. Verdicts will refresh nightly
                from SpendLogs and merged PRs.
              </p>
            </div>
          )}

          <DialogFooter className="flex-row justify-between sm:justify-between">
            <Button
              type="button"
              variant="outline"
              disabled={step === 0}
              onClick={() => setStep((current) => (current > 0 ? ((current - 1) as SetupStep) : current))}
            >
              Back
            </Button>
            {step < 3 ? (
              <Button
                type="button"
                disabled={step === 0 && !canContinue}
                onClick={() => setStep((current) => (current < 3 ? ((current + 1) as SetupStep) : current))}
              >
                Continue
              </Button>
            ) : (
              <Button type="button" onClick={finishSetup}>
                Finish
              </Button>
            )}
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}
