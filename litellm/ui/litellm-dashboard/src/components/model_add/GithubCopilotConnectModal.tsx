"use client";

import { useEffect, useReducer } from "react";
import { Trans, useTranslation } from "react-i18next";

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
  const { t } = useTranslation();
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
          <DialogTitle>{t("modelAdd.githubCopilotConnectModal.title")}</DialogTitle>
          <DialogDescription>
            {t("modelAdd.githubCopilotConnectModal.credential", { name: credentialName })}
          </DialogDescription>
        </DialogHeader>

        {state.kind === "starting" && (
          <p className="text-sm">{t("modelAdd.githubCopilotConnectModal.requestingCode")}</p>
        )}

        {state.kind === "awaiting" && (
          <div className="flex flex-col gap-3">
            <p className="text-sm">
              <Trans
                i18nKey="modelAdd.githubCopilotConnectModal.openAndEnterCode"
                components={{
                  verificationLink: (
                    <a
                      href={state.verificationUri}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-primary underline-offset-4 hover:underline"
                    >
                      {state.verificationUri}
                    </a>
                  ),
                }}
              />
            </p>
            <div className="flex items-center gap-2">
              <code data-testid="github-device-code" className="rounded-md bg-muted px-3 py-2 font-mono text-lg">
                {state.userCode}
              </code>
              <CopyButton
                value={state.userCode}
                label={t("modelAdd.githubCopilotConnectModal.copyCode")}
                copiedLabel={t("modelAdd.githubCopilotConnectModal.codeCopied")}
              />
            </div>
            <p className="text-sm text-muted-foreground">
              {t("modelAdd.githubCopilotConnectModal.waitingForApproval")}
            </p>
          </div>
        )}

        {state.kind === "connected" && (
          <p className="text-sm">{t("modelAdd.githubCopilotConnectModal.connectedAs", { login: state.githubLogin })}</p>
        )}

        {state.kind === "failed" && (
          <p role="alert" className="text-sm text-destructive">
            {failureMessage(state, t)}
          </p>
        )}

        <DialogFooter>
          {state.kind === "failed" && (
            <Button variant="outline" onClick={() => dispatch({ type: "restarted" })}>
              {t("modelAdd.githubCopilotConnectModal.startAgain")}
            </Button>
          )}
          {state.kind === "connected" && (
            <Button variant="destructive" onClick={onDisconnect}>
              {t("modelAdd.githubCopilotConnectModal.disconnect")}
            </Button>
          )}
          <Button onClick={onClose}>
            {state.kind === "connected"
              ? t("modelAdd.githubCopilotConnectModal.done")
              : t("modelAdd.githubCopilotConnectModal.close")}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
