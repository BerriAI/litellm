"use client";

import { useState } from "react";
import { ArrowLeft, ArrowRight, Check, CheckCircle2, Github, Gitlab, KeyRound, LockKeyhole } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "@/components/ui/dialog";

type Provider = "GitHub" | "GitLab";
const providerIcons = { GitHub: Github, GitLab: Gitlab };

type Step = "provider" | "authorize" | "repositories" | "complete";
const stepTitle: Record<Step, string> = {
  provider: "Connect your code",
  authorize: "Connect your code",
  repositories: "Choose your repositories",
  complete: "Ready to measure",
};

export default function ObservedConnections({ repo, onClose }: { repo: string; onClose: () => void }) {
  const [provider, setProvider] = useState<Provider>("GitHub");
  const [method, setMethod] = useState<"app" | "token">("app");
  const [step, setStep] = useState<Step>("provider");
  const [selected, setSelected] = useState(true);
  const [instance, setInstance] = useState("https://gitlab.com");
  const Icon = providerIcons[provider];
  const exampleRepo = provider === "GitHub" ? repo : "engineering/example-project";

  return (
    <Dialog
      open
      onOpenChange={(open) => {
        if (!open) onClose();
      }}
    >
      <DialogContent className="sm:max-w-lg">
        <DialogHeader>
          <div className="mb-2 flex items-center gap-2">
            <Badge variant="secondary">Setup preview</Badge>
            <span className="text-xs text-muted-foreground">No credentials are saved</span>
          </div>
          <DialogTitle>{stepTitle[step]}</DialogTitle>
          <DialogDescription>
            {step === "complete"
              ? "That is the proposed setup flow. Your report is still using the validated GitHub snapshot."
              : "Connect once. See shipping, quality, and recorded AI spend together."}
          </DialogDescription>
        </DialogHeader>
        {step === "provider" && (
          <div className="space-y-5 py-2">
            <div className="grid grid-cols-2 gap-3" role="group" aria-label="Code provider">
              {(["GitHub", "GitLab"] as const).map((name) => (
                <Button
                  key={name}
                  variant={provider === name ? "default" : "outline"}
                  className="h-12"
                  aria-pressed={provider === name}
                  onClick={() => setProvider(name)}
                >
                  {name === "GitHub" ? <Github /> : <Gitlab />}
                  {name}
                </Button>
              ))}
            </div>
            {provider === "GitLab" && (
              <div className="space-y-2">
                <label htmlFor="gitlab-instance" className="text-sm font-medium">
                  GitLab instance
                </label>
                <Input id="gitlab-instance" value={instance} onChange={(event) => setInstance(event.target.value)} />
                <p className="text-xs text-muted-foreground">GitLab.com or your self-managed instance</p>
              </div>
            )}
            <div className="space-y-2">
              <button
                type="button"
                aria-pressed={method === "app"}
                onClick={() => setMethod("app")}
                className={`flex w-full items-start gap-3 rounded-lg border p-4 text-left ${method === "app" ? "border-primary bg-primary/5" : "border-border"}`}
              >
                <Icon className="mt-0.5 size-5" />
                <span className="flex-1">
                  <span className="flex items-center gap-2 font-medium">
                    Connect with {provider} <Badge variant="secondary">Recommended</Badge>
                  </span>
                  <span className="mt-1 block text-xs text-muted-foreground">
                    {provider === "GitHub"
                      ? "Install the app and choose which repositories it can read"
                      : "Authorize the OAuth app with read-only access"}
                  </span>
                </span>
                {method === "app" && <Check className="size-4" />}
              </button>
              <button
                type="button"
                aria-pressed={method === "token"}
                onClick={() => setMethod("token")}
                className={`flex w-full items-start gap-3 rounded-lg border p-4 text-left ${method === "token" ? "border-primary bg-primary/5" : "border-border"}`}
              >
                <KeyRound className="mt-0.5 size-5" />
                <span className="flex-1">
                  <span className="font-medium">Use an access token</span>
                  <span className="mt-1 block text-xs text-muted-foreground">
                    Bring your own token with read access
                  </span>
                </span>
                {method === "token" && <Check className="size-4" />}
              </button>
            </div>
            <div className="flex items-center gap-2 text-xs text-muted-foreground">
              <LockKeyhole className="size-3.5" />
              Pull request and issue metadata. No source code or model required
            </div>
            <Button className="w-full" onClick={() => setStep("authorize")}>
              Preview {method === "app" ? `${provider} authorization` : "token setup"}
              <ArrowRight />
            </Button>
          </div>
        )}
        {step === "authorize" && (
          <div className="space-y-5 py-2">
            <div className="rounded-lg border p-5">
              <Icon className="mb-4 size-7" />
              <h3 className="font-medium">
                {method === "app" ? `Authorize LiteLLM on ${provider}` : `${provider} access token`}
              </h3>
              {method === "app" ? (
                <p className="mt-2 text-sm leading-relaxed text-muted-foreground">
                  In the live flow, {provider} opens to approve read access to{" "}
                  {provider === "GitHub"
                    ? "repository metadata, pull requests, and issues"
                    : "your profile and project API"}
                  . You return here to choose repositories.
                </p>
              ) : (
                <div className="mt-4 space-y-2">
                  <label htmlFor="example-token" className="text-sm">
                    Access token
                  </label>
                  <Input id="example-token" type="password" value="example-token-only" readOnly />
                  <p className="text-xs text-muted-foreground">
                    Example only. Do not enter a real token in this prototype
                  </p>
                </div>
              )}
            </div>
            <p className="text-xs leading-relaxed text-muted-foreground">
              This preview does not contact {provider}. The production flow requires{" "}
              {method === "app"
                ? "a configured app and callback URL"
                : "server-side token validation and encrypted storage"}
              .
            </p>
            <div className="flex justify-between">
              <Button variant="ghost" onClick={() => setStep("provider")}>
                <ArrowLeft />
                Back
              </Button>
              <Button onClick={() => setStep("repositories")}>
                Continue with example
                <ArrowRight />
              </Button>
            </div>
          </div>
        )}
        {step === "repositories" && (
          <div className="space-y-5 py-2">
            <label className="flex cursor-pointer items-center gap-3 rounded-lg border p-4">
              <input
                type="checkbox"
                checked={selected}
                onChange={(event) => setSelected(event.target.checked)}
                className="size-4 accent-primary"
              />
              <Icon className="size-4" />
              <span className="flex-1 text-sm font-medium">{exampleRepo}</span>
              <Badge variant="secondary">{provider === "GitHub" ? "Public" : "Example"}</Badge>
            </label>
            <div className="rounded-lg bg-muted/50 p-4 text-sm">
              <div className="flex items-center gap-2 font-medium">
                <CheckCircle2 className="size-4 text-emerald-600" />
                Gateway spend is already connected
              </div>
              <p className="mt-2 text-xs leading-relaxed text-muted-foreground">
                Match repository contributors to gateway users, then load merged PRs and issues. No estimator to
                configure
              </p>
            </div>
            <div className="flex justify-between">
              <Button variant="ghost" onClick={() => setStep("authorize")}>
                <ArrowLeft />
                Back
              </Button>
              <Button disabled={!selected} onClick={() => setStep("complete")}>
                Preview finish
                <ArrowRight />
              </Button>
            </div>
          </div>
        )}
        {step === "complete" && (
          <div className="space-y-5 py-3">
            <div className="rounded-lg border p-5">
              <CheckCircle2 className="mb-3 size-8 text-emerald-600" />
              <p className="font-medium">
                {provider} via {method === "app" ? "app" : "token"}
              </p>
              <p className="mt-1 text-sm text-muted-foreground">{exampleRepo}</p>
              <p className="mt-4 text-xs text-muted-foreground">Preview complete. No new connection was created</p>
            </div>
            <Button className="w-full" onClick={onClose}>
              Back to report
            </Button>
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
}
