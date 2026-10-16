import { useState } from "react";
import { Inbox } from "lucide-react";

import { useTernarySettings } from "@/app/(dashboard)/hooks/ternary/useTernary";
import { Alert, AlertDescription, AlertTitle } from "@/components/shared/Alert";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";

import { TernaryIntegrationSettings } from "./TernaryIntegrationSettings";
import TernarySettingsModal from "./TernarySettingsModal";

export default function TernaryCostTracking() {
  const { data: settings, isLoading, error } = useTernarySettings();
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [restartNotice, setRestartNotice] = useState<string | null>(null);

  if (isLoading) {
    return (
      <Card>
        <CardContent>
          <p className="text-sm text-muted-foreground">Loading Ternary settings...</p>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card>
        <CardContent>
          <p className="text-sm text-destructive">Error loading Ternary settings: {error.message}</p>
        </CardContent>
      </Card>
    );
  }

  const current = settings ?? null;

  const handleSaved = (message: string) => {
    setIsModalOpen(false);
    setRestartNotice(message);
  };

  return (
    <>
      {restartNotice && (
        <Alert className="mx-auto mb-6 max-w-4xl">
          <AlertTitle>Restart the proxy to apply</AlertTitle>
          <AlertDescription>{restartNotice}</AlertDescription>
        </Alert>
      )}

      {current ? (
        <TernaryIntegrationSettings
          settings={current}
          onEdit={() => setIsModalOpen(true)}
          onDeleted={setRestartNotice}
        />
      ) : (
        <div className="mx-auto mt-8 max-w-2xl rounded-lg border border-dashed border-border bg-card p-12 text-center">
          <div className="flex flex-col items-center gap-2">
            <Inbox className="size-10 text-muted-foreground" aria-hidden />
            <h4 className="text-base font-semibold">No Ternary Integration Found</h4>
            <p className="mx-auto max-w-md text-sm text-muted-foreground">
              Send LiteLLM spend to Ternary as FOCUS data, so AI costs can be allocated across teams.
            </p>
            <Button size="lg" onClick={() => setIsModalOpen(true)} className="mt-4">
              Add Ternary Integration
            </Button>
          </div>
        </div>
      )}

      <TernarySettingsModal
        key={current ? "edit" : "create"}
        open={isModalOpen}
        settings={current}
        onSaved={handleSaved}
        onCancel={() => setIsModalOpen(false)}
      />
    </>
  );
}
