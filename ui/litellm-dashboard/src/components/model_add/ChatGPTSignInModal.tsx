"use client";

import { useEffect, useReducer } from "react";

import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { chatGPTLoginPollCall, chatGPTLoginStartCall, type ChatGPTLoginStartRequest } from "@/components/networking";
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

import { initialSignInFlowState, signInFailureMessage, signInFlowReducer } from "./chatgpt_sign_in_flow";

interface ChatGPTSignInModalProps {
  readonly credentialName: string;
  readonly request: ChatGPTLoginStartRequest;
  readonly onClose: () => void;
  readonly onConnected: () => void;
}

export default function ChatGPTSignInModal({ credentialName, request, onClose, onConnected }: ChatGPTSignInModalProps) {
  const { accessToken } = useAuthorized();
  const [state, dispatch] = useReducer(signInFlowReducer, initialSignInFlowState);

  useEffect(() => {
    if (state.kind !== "starting" || !accessToken) {
      return;
    }
    let cancelled = false;
    chatGPTLoginStartCall(accessToken, credentialName, request)
      .then((response) => !cancelled && dispatch({ type: "started", response, now: Date.now() }))
      .catch((error: unknown) => !cancelled && dispatch({ type: "errored", message: extractProxyErrorMessage(error) }));
    return () => {
      cancelled = true;
    };
  }, [state, accessToken, credentialName, request]);

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
      chatGPTLoginPollCall(accessToken, credentialName, state.flowHandle)
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
          <DialogTitle>Sign in with ChatGPT</DialogTitle>
          <DialogDescription>Credential: {credentialName}</DialogDescription>
        </DialogHeader>

        {state.kind === "starting" && <p className="text-sm">Requesting a sign-in code from ChatGPT...</p>}

        {state.kind === "awaiting" && (
          <div className="flex flex-col gap-3">
            <p className="text-sm">
              Open{" "}
              <a
                href={state.verificationUri}
                target="_blank"
                rel="noopener noreferrer"
                className="text-primary underline-offset-4 hover:underline"
              >
                {state.verificationUri}
              </a>{" "}
              and enter this code:
            </p>
            <div className="flex items-center gap-2">
              <code data-testid="chatgpt-user-code" className="rounded-md bg-muted px-3 py-2 font-mono text-lg">
                {state.userCode}
              </code>
              <CopyButton value={state.userCode} label="Copy code" copiedLabel="Code copied" />
            </div>
            <p className="text-sm text-muted-foreground">Waiting for you to finish signing in on ChatGPT...</p>
          </div>
        )}

        {state.kind === "failed" && (
          <p role="alert" className="text-sm text-destructive">
            {signInFailureMessage(state)}
          </p>
        )}

        <DialogFooter>
          {state.kind === "failed" && (
            <Button variant="outline" onClick={() => dispatch({ type: "restarted" })}>
              Start again
            </Button>
          )}
          <Button onClick={onClose}>Close</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
