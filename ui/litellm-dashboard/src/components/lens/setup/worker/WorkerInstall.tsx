"use client";

import { Button } from "@/components/ui/button";

import { CheckCircle2, Copy, Loader2 } from "lucide-react";
import type { WorkerCreated } from "../../model/types";

import { workerSetupCommand } from "./workerCommand";
export function WorkerInstall({
  connected,
  address,
  created,
  copied,
  setCopied,
  setError,
  onReady,
  onClose,
}: {
  connected: boolean;
  address: string;
  created: WorkerCreated;
  copied: boolean;
  setCopied: (value: boolean) => void;
  setError: (message: string) => void;
  onReady?: () => void;
  onClose: () => void;
}) {
  return (
    <div className="min-w-0 space-y-5">
      {!connected && (
        <>
          <Button
            variant="default"
            className="w-full gap-2"
            onClick={async () => {
              try {
                await navigator.clipboard.writeText(workerSetupCommand(address, created.token));
                setCopied(true);
              } catch {
                setError("Clipboard access failed. Allow clipboard access and try again.");
              }
            }}
          >
            {copied ? <CheckCircle2 className="size-4" /> : <Copy className="size-4" />}
            {copied ? "Copied" : "Copy Docker command"}
          </Button>
          <details className="text-sm">
            <summary className="cursor-pointer text-muted-foreground">View command</summary>
            <p className="mt-3 text-xs text-muted-foreground">Contains a private worker token.</p>
            <pre
              aria-label="Docker command preview"
              className="mt-3 max-h-48 overflow-auto rounded-md bg-muted/40 p-3 text-xs leading-5"
            >
              {workerSetupCommand(address, created.token)}
            </pre>
          </details>
        </>
      )}
      {!connected && (
        <div className="space-y-3 border-t pt-5">
          <div role="status" className="flex items-center gap-2.5 text-sm">
            <Loader2 className="size-4 shrink-0 animate-spin text-muted-foreground" /> Waiting for your worker to
            connect…
          </div>
          <details className="pl-6.5 text-sm text-muted-foreground">
            <summary className="cursor-pointer">Not connecting?</summary>
            <p className="mt-2 break-words leading-6">
              Check that Docker is running and can reach {address}. Inspect the container logs for connection or
              authentication errors. This page updates automatically.
            </p>
          </details>
        </div>
      )}
      {connected && (
        <Button className="w-full" onClick={onReady ?? onClose}>
          {onReady ? "New investigation" : "Done"}
        </Button>
      )}
    </div>
  );
}
