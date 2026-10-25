"use client";

import { useLanguage } from "@/contexts/LanguageContext";
import { useEffect, useReducer } from "react";

import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { userConnectionPollCall, userConnectionStartCall } from "@/components/networking";
import CopyButton from "@/components/shared/CopyButton";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { extractProxyErrorMessage } from "@/lib/http/client";

import { connectFlowReducer, failureMessage, initialConnectFlowState } from "./github_copilot_connect_flow";

interface GithubCopilotConnectModalProps {
  readonly credentialName: string;
  readonly onClose: () => void;
  readonly onConnected: () => void;
  readonly onDisconnect: () => void;
}

export default function GithubCopilotConnectModal({
  credentialName,
  onClose,
  onConnected,
  onDisconnect,
}: GithubCopilotConnectModalProps) {
  const { t } = useLanguage();
  const { accessToken } = useAuthorized();
  const [state, dispatch] = useReducer(connectFlowReducer, initialConnectFlowState);

  useEffect(() => {
    if (state.kind !== "starting" || !accessToken) {
      return;
    }
    let cancelled = false;
    userConnectionStartCall(accessToken, credentialName)
      .then((response) => !cancelled && dispatch({ type: "started", response, now: Date.now() }))
      .catch((error: unknown) => !cancelled && dispatch({ type: "errored", message: extractProxyErrorMessage(error) }));
    return () => {
      cancelled = true;
    };
  }, [state, accessToken, credentialName]);

  useEffect(() => {
    if (state.kind !== "awaiting" || !accessToken) {
      return;
    }
    let cancelled = false;
    const timer = setTimeout(() => {
      if (Date.now() >= state.expiresAt) {
        dispatch({ type: "timed_out" });
        return;
      }
      userConnectionPollCall(accessToken, credentialName, state.flowHandle)
        .then((response) => !cancelled && dispatch({ type: "polled", response }))
        .catch(
          (error: unknown) => !cancelled && dispatch({ type: "errored", message: extractProxyErrorMessage(error) }),
        );
    }, state.intervalSeconds * 1000);
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [state, accessToken, credentialName]);

  useEffect(() => {
    if (state.kind === "connected") {
      onConnected();
    }
  }, [state.kind, onConnected]);

  return (
    <Dialog open onOpenChange={(open) => !open && onClose()}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>{t("Connect GitHub Copilot")}</DialogTitle>
          <DialogDescription>
            {t("Credential:")} {credentialName}
          </DialogDescription>
        </DialogHeader>

        {state.kind === "starting" && <p className="text-sm">{t("Requesting a device code from GitHub...")}</p>}

        {state.kind === "awaiting" && (
          <div className="flex flex-col gap-3">
            <p className="text-sm">
              {t("Open")}{" "}
              <a
                href={state.verificationUri}
                target="_blank"
                rel="noopener noreferrer"
                className="text-primary underline-offset-4 hover:underline"
              >
                {state.verificationUri}
              </a>{" "}
              {t("and enter this code:")}
            </p>
            <div className="flex items-center gap-2">
              <code data-testid="github-device-code" className="rounded-md bg-muted px-3 py-2 font-mono text-lg">
                {state.userCode}
              </code>
              <CopyButton value={state.userCode} label={t("Copy code")} copiedLabel="Code copied" />
            </div>
            <p className="text-sm text-muted-foreground">{t("Waiting for you to approve the request on GitHub...")}</p>
          </div>
        )}

        {state.kind === "connected" && (
          <p className="text-sm">{t("Connected as @{name}", { name: state.githubLogin })}</p>
        )}

        {state.kind === "failed" && (
          <p role="alert" className="text-sm text-destructive">
            {failureMessage(state)}
          </p>
        )}

        <DialogFooter>
          {state.kind === "failed" && (
            <Button variant="outline" onClick={() => dispatch({ type: "restarted" })}>
              {t("Start again")}
            </Button>
          )}
          {state.kind === "connected" && (
            <Button variant="destructive" onClick={onDisconnect}>
              {t("Disconnect")}
            </Button>
          )}
          <Button onClick={onClose}>{state.kind === "connected" ? t("Done") : t("Close")}</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
