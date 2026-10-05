"use client";

import { useEffect, useState } from "react";
import { Github, Gitlab, ArrowLeft, KeyRound } from "lucide-react";
import { z } from "zod";
import { apiClient } from "@/components/networking";
import { extractProxyErrorMessage } from "@/lib/http/client";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "@/components/ui/dialog";
import {
  observedSettingsSchema,
  repositoryNames,
  type ObservedSettings,
  type ObservedConnection,
} from "./observedData";

const appFields = {
  configured: z.boolean(),
  can_install: z.boolean().optional().default(false),
  api_url: z.string().nullable(),
  callback_url: z.string().nullable(),
};
const appSchema = z.object(appFields);
const appsSchema = z.object({ github: appSchema, gitlab: appSchema });
const repositoriesSchema = z.object({
  repositories: z.array(z.object({ name: z.string(), visibility: z.string(), archived: z.boolean() })),
  has_more: z.boolean(),
});
const defaultUrl = { github: "https://api.github.com", gitlab: "https://gitlab.com/api/v4" };

function preferredConnectionMethod(selected: "app" | "token" | null, configured: boolean | undefined) {
  return selected ?? (configured ? "app" : "token");
}

function TokenFields({
  label,
  provider,
  token,
  onToken,
  apiUrl,
  onUrl,
  hasToken,
}: {
  label: string;
  provider: ObservedSettings["source_provider"];
  token: string;
  onToken: (value: string) => void;
  apiUrl: string;
  onUrl: (value: string) => void;
  hasToken: boolean;
}) {
  return (
    <>
      <label htmlFor="roi-source-token" className="block text-sm font-medium">
        {label} access token
      </label>
      <Input
        id="roi-source-token"
        type="password"
        autoComplete="off"
        value={token}
        onChange={(event) => onToken(event.target.value)}
        placeholder={hasToken ? "Leave blank to keep the saved token" : "Optional for public repositories"}
      />
      <p className="text-xs text-muted-foreground">
        {provider === "github"
          ? "Fine-grained token: read access to pull requests, issues, and metadata"
          : "Token with read_api scope"}
      </p>
      <details className="text-sm">
        <summary className="cursor-pointer">Self-hosted instance</summary>
        <label htmlFor="roi-source-url" className="mt-3 block text-xs">
          API URL
        </label>
        <Input id="roi-source-url" value={apiUrl} onChange={(event) => onUrl(event.target.value)} />
      </details>
    </>
  );
}
function AppMessage({ configured, label }: { configured: boolean; label: string }) {
  return (
    <p className="text-sm text-muted-foreground">
      {configured
        ? `You’ll authorize ${label}, then choose repositories`
        : `Register the ${label} app in gateway settings, or connect with a token`}
    </p>
  );
}

function RepositoryChoices({
  available,
  repos,
  setRepos,
  query,
  setQuery,
  page,
  setPage,
}: {
  available: z.infer<typeof repositoriesSchema> | null;
  repos: string;
  setRepos: (value: string) => void;
  query: string;
  setQuery: (value: string) => void;
  page: number;
  setPage: (value: number) => void;
}) {
  return (
    <>
      <Input
        aria-label="Find repositories"
        value={query}
        onChange={(event) => {
          setQuery(event.target.value);
          setPage(1);
        }}
        placeholder="Find repositories…"
      />
      <div className="max-h-48 overflow-y-auto rounded-lg border divide-y">
        {!available && (
          <p role="status" className="p-3 text-sm text-muted-foreground">
            Loading repositories…
          </p>
        )}
        {available?.repositories.length === 0 && (
          <p className="p-3 text-sm text-muted-foreground">No repositories found</p>
        )}
        {available?.repositories
          .filter((repo) => !repo.archived)
          .map((repo) => (
            <label key={repo.name} className="flex cursor-pointer items-center gap-3 p-3 text-sm">
              <input
                type="checkbox"
                checked={repositoryNames(repos).includes(repo.name)}
                onChange={(event) =>
                  setRepos(
                    (event.target.checked
                      ? [...new Set([...repositoryNames(repos), repo.name])]
                      : repositoryNames(repos).filter((name) => name !== repo.name)
                    ).join(", "),
                  )
                }
              />
              <span className="flex-1">{repo.name}</span>
              <span className="text-xs text-muted-foreground">{repo.visibility}</span>
            </label>
          ))}
      </div>
      <div className="flex justify-between">
        <Button size="sm" variant="ghost" disabled={page === 1} onClick={() => setPage(page - 1)}>
          Previous
        </Button>
        <Button size="sm" variant="ghost" disabled={!available?.has_more} onClick={() => setPage(page + 1)}>
          Next
        </Button>
      </div>
    </>
  );
}

function ConnectionMethod({
  method,
  setMethod,
  label,
  provider,
  token,
  setToken,
  apiUrl,
  setApiUrl,
  connected,
  apps,
  busy,
  connect,
}: {
  method: "app" | "token";
  setMethod: (value: "app" | "token") => void;
  label: string;
  provider: ObservedSettings["source_provider"];
  token: string;
  setToken: (value: string) => void;
  apiUrl: string;
  setApiUrl: (value: string) => void;
  connected: ObservedConnection;
  apps: z.infer<typeof appsSchema> | null;
  busy: boolean;
  connect: () => void;
}) {
  const connectLabel = method === "app" ? `Connect ${label}` : "Continue";
  const sameSource = connected.source_provider === provider && connected.api_url === apiUrl;
  const hasSavedToken = sameSource && connected.has_token && connected.connection_type === "token";
  return (
    <>
      <div className="grid grid-cols-2 gap-2">
        <Button
          variant={method === "app" ? "secondary" : "outline"}
          aria-pressed={method === "app"}
          onClick={() => setMethod("app")}
        >
          Connect with app
        </Button>
        <Button
          variant={method === "token" ? "secondary" : "outline"}
          aria-pressed={method === "token"}
          onClick={() => setMethod("token")}
        >
          <KeyRound />
          Access token
        </Button>
      </div>
      {method === "token" && (
        <TokenFields
          label={label}
          provider={provider}
          token={token}
          onToken={setToken}
          apiUrl={apiUrl}
          onUrl={setApiUrl}
          hasToken={hasSavedToken}
        />
      )}
      {method === "app" && <AppMessage configured={Boolean(apps?.[provider].configured)} label={label} />}
      <Button
        className="w-full"
        disabled={busy || (method === "app" && !apps?.[provider].configured)}
        onClick={connect}
      >
        {busy ? "Connecting…" : connectLabel}
      </Button>
    </>
  );
}

type ConnectionStep = "list" | "connect" | "repos";
const stepTitles = { list: "Connections", connect: "Connect your code", repos: "Choose repositories" };

function initialStep(settings: ObservedSettings, afterAuthorization: boolean): ConnectionStep {
  if (afterAuthorization) return "repos";
  if (settings.connections?.length) return "list";
  return settings.has_token || settings.ready ? "repos" : "connect";
}

function stepDescription(step: ConnectionStep, label: string) {
  if (step === "list") return "All selected repositories appear in one report";
  if (step === "connect") return "Connect GitHub and GitLab with an app or access token";
  return `Select ${label} repositories to compare`;
}

function connectionMethodLabel(entry: ObservedConnection) {
  if (entry.connection_type === "app") return "App";
  return entry.has_token ? "Token" : "Public access";
}

function ConnectionList({
  connections,
  onEdit,
  onAdd,
}: {
  connections: ObservedConnection[];
  onEdit: (entry: ObservedConnection) => void;
  onAdd: () => void;
}) {
  return (
    <>
      {connections.map((entry) => (
        <div key={entry.id} className="flex items-center justify-between gap-3 rounded-lg border p-4">
          <div className="min-w-0 text-sm">
            <p className="flex items-center gap-2 font-medium">
              {entry.source_provider === "github" ? <Github className="size-4" /> : <Gitlab className="size-4" />}
              {entry.source_provider === "github" ? "GitHub" : "GitLab"}
            </p>
            <p className="mt-1 truncate text-xs text-muted-foreground">{new URL(entry.api_url).host}</p>
            <p className="mt-2 text-xs text-muted-foreground">
              {entry.repos.length} repositories · {connectionMethodLabel(entry)}
            </p>
          </div>
          <Button
            variant="outline"
            size="sm"
            onClick={() => onEdit(entry)}
            aria-label={`Edit ${entry.source_provider === "github" ? "GitHub" : "GitLab"} ${new URL(entry.api_url).host}`}
          >
            Edit
          </Button>
        </div>
      ))}
      <Button className="w-full" variant="outline" onClick={onAdd}>
        Add connection
      </Button>
    </>
  );
}

function initialMethod(settings: ObservedSettings) {
  return settings.has_token ? settings.connection_type : null;
}

function hasConnections(settings: ObservedSettings) {
  return Boolean(settings.connections?.length);
}

function canManageApp(connected: ObservedConnection, apps: z.infer<typeof appsSchema> | null) {
  return (
    connected.connection_type === "app" && connected.source_provider === "github" && Boolean(apps?.github.can_install)
  );
}

export default function ObservedConnections({
  accessToken,
  settings,
  onClose,
  onSaved,
  initialError = "",
  afterAuthorization = false,
}: {
  accessToken: string;
  settings: ObservedSettings;
  onClose: () => void;
  onSaved: () => void;
  initialError?: string;
  afterAuthorization?: boolean;
}) {
  const [savedSettings, setSavedSettings] = useState(settings);
  const [connected, setConnected] = useState<ObservedConnection>(settings);
  const [provider, setProvider] = useState(settings.source_provider);
  const [apiUrl, setApiUrl] = useState(settings.api_url);
  const [selectedMethod, setMethod] = useState<"app" | "token" | null>(initialMethod(settings));
  const [step, setStep] = useState<ConnectionStep>(() => initialStep(settings, afterAuthorization));
  const [token, setToken] = useState("");
  const [repos, setRepos] = useState(settings.repos.join(", "));
  const [apps, setApps] = useState<z.infer<typeof appsSchema> | null>(null);
  const method = preferredConnectionMethod(selectedMethod, apps?.[provider].configured);
  const [available, setAvailable] = useState<z.infer<typeof repositoriesSchema> | null>(null);
  const [query, setQuery] = useState("");
  const [page, setPage] = useState(1);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(initialError);
  const label = provider === "github" ? "GitHub" : "GitLab";
  const saveLabel = repositoryNames(repos).length ? "Save and sync" : "Save repositories";
  const manageApp = canManageApp(connected, apps);
  const showConnections = hasConnections(savedSettings);
  useEffect(() => {
    const controller = new AbortController();
    apiClient
      .get<unknown>("/roi-calculator/observed/apps", { accessToken, signal: controller.signal })
      .then((data) => {
        if (!controller.signal.aborted) setApps(appsSchema.parse(data));
      })
      .catch((reason: unknown) => {
        if (!controller.signal.aborted) setError(extractProxyErrorMessage(reason));
      });
    return () => controller.abort();
  }, [accessToken]);
  useEffect(() => {
    if (step !== "repos" || (!connected.has_token && connected.source_provider === "github")) return;
    const controller = new AbortController();
    const timer = setTimeout(() => {
      apiClient
        .get<unknown>("/roi-calculator/observed/repositories", {
          accessToken,
          signal: controller.signal,
          query: { query, page, connection: connected.id },
        })
        .then((data) => {
          if (!controller.signal.aborted) setAvailable(repositoriesSchema.parse(data));
        })
        .catch((reason: unknown) => {
          if (!controller.signal.aborted) setError(extractProxyErrorMessage(reason));
        });
    }, 250);
    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [accessToken, step, connected, query, page]);
  function selectProvider(value: ObservedSettings["source_provider"]) {
    setProvider(value);
    const existing = savedSettings.connections?.find(
      (entry) => entry.source_provider === value && entry.api_url === defaultUrl[value],
    );
    setApiUrl(existing?.api_url ?? defaultUrl[value]);
    setConnected(
      existing ?? {
        ...settings,
        source_provider: value,
        api_url: defaultUrl[value],
        repos: [],
        has_token: false,
        ready: false,
        connection_type: "token",
        id: undefined,
      },
    );
    setMethod(existing?.connection_type ?? null);
    setToken("");
    setError("");
  }
  function searchRepositories(value: string) {
    setAvailable(null);
    setError("");
    setQuery(value);
  }
  function changePage(value: number) {
    setAvailable(null);
    setError("");
    setPage(value);
  }
  async function connect(install = false) {
    setBusy(true);
    setError("");
    try {
      if (method === "app" || install) {
        const sameApp = connected.source_provider === provider && connected.connection_type === "app";
        const firstInstallation = provider === "github" && !sameApp && apps?.github.can_install;
        const result = z.object({ url: z.string().url() }).parse(
          await apiClient.post<unknown>(`/roi-calculator/observed/oauth/${provider}/start`, {
            accessToken,
            credentials: "include",
            query: { install: install || Boolean(firstInstallation) },
          }),
        );
        window.location.assign(result.url);
        return;
      }
      const same = provider === connected.source_provider && apiUrl === connected.api_url;
      const keepToken = same && connected.has_token && connected.connection_type === "token";
      const result = observedSettingsSchema.parse(
        await apiClient.put<unknown>("/roi-calculator/observed/settings", {
          accessToken,
          body: {
            connection_id: savedSettings.connections?.find((entry) => entry.id === connected.id)?.id,
            source_provider: provider,
            api_url: apiUrl,
            token: token || (keepToken ? undefined : ""),
            repos: same ? connected.repos : [],
            update_interval_minutes: connected.update_interval_minutes,
          },
        }),
      );
      setSavedSettings(result);
      setConnected(result);
      setToken("");
      setRepos(result.repos.join(", "));
      setAvailable(null);
      setStep("repos");
    } catch (reason) {
      setError(extractProxyErrorMessage(reason));
    } finally {
      setBusy(false);
    }
  }
  async function save() {
    setBusy(true);
    setError("");
    try {
      const result = observedSettingsSchema.parse(
        await apiClient.put<unknown>("/roi-calculator/observed/settings", {
          accessToken,
          body: {
            connection_id: connected.id,
            source_provider: connected.source_provider,
            api_url: connected.api_url,
            repos: repositoryNames(repos),
            update_interval_minutes: connected.update_interval_minutes,
          },
        }),
      );
      if (result.ready) await apiClient.post<unknown>("/roi-calculator/observed/sync", { accessToken });
      onSaved();
      onClose();
    } catch (reason) {
      setError(extractProxyErrorMessage(reason));
    } finally {
      setBusy(false);
    }
  }
  function edit(entry: ObservedConnection) {
    setConnected(entry);
    setProvider(entry.source_provider);
    setApiUrl(entry.api_url);
    setMethod(entry.connection_type);
    setRepos(entry.repos.join(", "));
    setToken("");
    setQuery("");
    setPage(1);
    setAvailable(null);
    setError("");
    setStep("repos");
  }
  return (
    <Dialog
      open
      onOpenChange={(open) => {
        if (!open) onClose();
      }}
    >
      <DialogContent className="sm:max-w-lg">
        <DialogHeader>
          <DialogTitle>{stepTitles[step]}</DialogTitle>
          <DialogDescription>{stepDescription(step, label)}</DialogDescription>
        </DialogHeader>
        <div className="space-y-4 py-2">
          {step === "list" && (
            <ConnectionList
              connections={savedSettings.connections ?? []}
              onEdit={edit}
              onAdd={() => {
                selectProvider(
                  savedSettings.connections?.some((entry) => entry.source_provider === "github") ? "gitlab" : "github",
                );
                setStep("connect");
              }}
            />
          )}
          {step === "connect" && (
            <>
              {showConnections && (
                <Button variant="ghost" size="sm" onClick={() => setStep("list")}>
                  <ArrowLeft />
                  All connections
                </Button>
              )}
              <div className="grid grid-cols-2 gap-3" role="group" aria-label="Code provider">
                {(["github", "gitlab"] as const).map((value) => (
                  <Button
                    key={value}
                    variant={provider === value ? "default" : "outline"}
                    aria-pressed={provider === value}
                    onClick={() => selectProvider(value)}
                  >
                    {value === "github" ? <Github /> : <Gitlab />}
                    {value === "github" ? "GitHub" : "GitLab"}
                  </Button>
                ))}
              </div>
              <ConnectionMethod
                method={method}
                setMethod={setMethod}
                label={label}
                provider={provider}
                token={token}
                setToken={setToken}
                apiUrl={apiUrl}
                setApiUrl={setApiUrl}
                connected={connected}
                apps={apps}
                busy={busy}
                connect={() => connect()}
              />
            </>
          )}
          {step === "repos" && (
            <>
              {showConnections && (
                <Button variant="ghost" size="sm" onClick={() => setStep("list")}>
                  <ArrowLeft />
                  All connections
                </Button>
              )}
              <Button variant="ghost" size="sm" onClick={() => setStep("connect")}>
                <ArrowLeft />
                Change connection
              </Button>
              {manageApp && (
                <Button variant="outline" size="sm" disabled={busy} onClick={() => connect(true)}>
                  Manage GitHub repositories
                </Button>
              )}
              <label htmlFor="roi-repositories" className="block text-sm font-medium">
                Repositories
              </label>
              <Input
                id="roi-repositories"
                value={repos}
                onChange={(event) => setRepos(event.target.value)}
                placeholder={
                  provider === "github" ? "owner/repo, owner/another-repo" : "group/project, group/subgroup/project"
                }
              />
              {(connected.has_token || connected.source_provider === "gitlab") && (
                <>
                  <RepositoryChoices
                    available={available}
                    repos={repos}
                    setRepos={setRepos}
                    query={query}
                    setQuery={searchRepositories}
                    page={page}
                    setPage={changePage}
                  />
                </>
              )}
              <Button className="w-full" disabled={busy} onClick={save}>
                {busy ? "Saving…" : saveLabel}
              </Button>
            </>
          )}
          {error && (
            <p role="alert" className="text-sm text-destructive">
              {error}
            </p>
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
}
