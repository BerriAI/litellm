"use client";

import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { AnalysisKeyPicker } from "./AnalysisKeyPicker";

import { AnalysisAccessFields } from "./AnalysisAccessFields";
import type { AnalysisAccess } from "./workerSchema";
export function WorkerForm({
  useExisting,
  accessToken,
  analysisKey,
  setAnalysisKey,
  access,
  setAccess,
  editingWorker,
  setUseExisting,
  address,
  setAddress,
}: {
  useExisting: boolean;
  accessToken: string;
  analysisKey: string | null;
  setAnalysisKey: (key: string | null) => void;
  access: AnalysisAccess;
  setAccess: (value: AnalysisAccess) => void;
  editingWorker: string | null;
  setUseExisting: (value: boolean) => void;
  address: string;
  setAddress: (value: string) => void;
}) {
  return (
    <div className="min-w-0 space-y-5">
      {useExisting ? (
        <AnalysisKeyPicker accessToken={accessToken} value={analysisKey} onChange={setAnalysisKey} />
      ) : (
        <AnalysisAccessFields
          accessToken={accessToken}
          value={access}
          onChange={(next) => {
            setAccess(next);
          }}
        />
      )}
      <details className="text-sm" open={editingWorker ? true : undefined}>
        <summary className="cursor-pointer font-medium">Advanced options</summary>
        <div className="mt-4 space-y-5">
          <label className="flex items-center justify-between gap-4">
            Use an existing virtual key
            <Switch checked={useExisting} onCheckedChange={setUseExisting} />
          </label>
          {!editingWorker && (
            <div className="space-y-2">
              <label htmlFor="worker-proxy-address" className="block text-sm font-medium">
                LiteLLM proxy URL
              </label>
              <Input id="worker-proxy-address" value={address} onChange={(e) => setAddress(e.target.value)} />
              <p className="text-xs text-muted-foreground">Your server must be able to reach this address.</p>
            </div>
          )}
        </div>
      </details>
    </div>
  );
}
