"use client";

import { useId, useState } from "react";
import { Loader2, XIcon } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Dialog, DialogContent, DialogDescription, DialogTitle } from "@/components/ui/dialog";
import { Label } from "@/components/ui/label";
import { useStoredValue } from "@/lib/storage";
import { useLensReadiness } from "../hooks/useLensReadiness";
import { LENS_INTRO_DISMISSED, LENS_INTRO_SEEN } from "../storage";
import { LensGettingStarted, type LensGettingStartedProps } from "./LensGettingStarted";
import { useOnboarding } from "./OnboardingContext";

export interface LensIntro {
  readonly open: boolean;
  close(forever: boolean): void;
}

/** Opens on the first Lens visit of a session, or whenever the URL asks for setup. "Don't show this again" outlives the tab; a plain close only rests for the session. */
export function useLensIntro({ demo, settingUp }: { demo: boolean; settingUp: boolean }): LensIntro {
  const [dismissed, setDismissed] = useStoredValue(LENS_INTRO_DISMISSED);
  const [seen, setSeen] = useStoredValue(LENS_INTRO_SEEN);
  const close = (forever: boolean) => {
    setSeen(true);
    if (forever) setDismissed(true);
  };
  const firstVisit = !demo && !dismissed && !seen;
  return { open: settingUp || firstVisit, close };
}

export type LensIntroDialogProps = Omit<LensGettingStartedProps, "state"> & {
  open: boolean;
  onClose: (forever: boolean) => void;
};

export function LensIntroDialog({ open, onClose, ...gettingStarted }: LensIntroDialogProps) {
  const [forever, setForever] = useState(false);
  const checkboxId = useId();
  const close = () => onClose(forever);
  return (
    <Dialog open={open} onOpenChange={(next) => !next && close()}>
      <DialogContent
        showCloseButton={false}
        className="max-h-[calc(100vh-4rem)] gap-0 overflow-y-auto p-0 sm:max-w-4xl xl:max-w-5xl 2xl:max-w-6xl"
      >
        <div className="sticky top-0 z-raised flex items-center justify-end gap-3 bg-popover/90 px-4 py-2 backdrop-blur-sm">
          <Label htmlFor={checkboxId} className="gap-2 text-xs font-normal text-muted-foreground">
            <Checkbox id={checkboxId} checked={forever} onCheckedChange={(checked) => setForever(checked === true)} />
            Don’t show this again
          </Label>
          <Button variant="ghost" size="icon-sm" aria-label="Close" onClick={close}>
            <XIcon />
          </Button>
        </div>
        <DialogTitle className="sr-only">Get started with Lens</DialogTitle>
        <DialogDescription className="sr-only">
          What Lens does, and the steps to connect tracing, a worker and your first investigation.
        </DialogDescription>
        <div className="px-5 pb-5">
          <IntroContent {...gettingStarted} />
        </div>
      </DialogContent>
    </Dialog>
  );
}

function IntroContent(props: Omit<LensGettingStartedProps, "state">) {
  const { canViewInvestigations } = useOnboarding();
  const state = useLensReadiness(canViewInvestigations);
  if (state.loading)
    return (
      <p role="status" className="flex items-center gap-2 py-8 text-sm text-muted-foreground">
        <Loader2 aria-hidden="true" className="size-4 animate-spin" />
        Checking Lens setup…
      </p>
    );
  return <LensGettingStarted state={state} {...props} />;
}
