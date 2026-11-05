"use client";

import { Mail } from "lucide-react";
import { Button, buttonVariants } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { uiHref } from "@/utils/uiHref";

const emailSettingsHref = `${uiHref("logging-and-alerts")}?tab=email-alerts`;
const setupSteps = [
  {
    title: "Connect email delivery",
    description:
      "Add your SMTP settings and public gateway URL in Email Settings. Send a test email to verify delivery.",
  },
  {
    title: "Enable model deprecation emails",
    description: "In Email events, turn on Model Deprecations and save. This is off by default.",
  },
  {
    title: "Owners get two reminders",
    description:
      "Affected key owners and team admins receive a reminder 30 days before retirement, and one final reminder 1 day before. Recipients are based on actual usage in the last 30 days. Dates are checked in UTC.",
  },
];

export default function DeprecationEmailSetup() {
  return (
    <Dialog>
      <div className="flex flex-wrap items-center justify-between gap-2 border-b pb-3 text-xs text-muted-foreground">
        <span className="flex items-center gap-2">
          <Mail className="size-3.5 shrink-0" aria-hidden />
          Give affected owners time to migrate. Email reminders at 30 days and 1 day.
        </span>
        <DialogTrigger render={<Button variant="link" size="sm" />}>Set up email reminders</DialogTrigger>
      </div>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Set up model deprecation emails</DialogTitle>
          <DialogDescription>
            Help key owners and team admins act before a provider retires a model. Only their affected usage is
            included.
          </DialogDescription>
        </DialogHeader>
        <ol className="my-3 space-y-5">
          {setupSteps.map((step, index) => (
            <li key={step.title} className="flex gap-3">
              <span
                className="flex size-7 shrink-0 items-center justify-center rounded-full border text-xs font-medium"
                aria-hidden
              >
                {index + 1}
              </span>
              <div>
                <h3 className="text-sm font-medium">{step.title}</h3>
                <p className="mt-1 text-sm leading-relaxed text-muted-foreground">{step.description}</p>
              </div>
            </li>
          ))}
        </ol>
        <DialogFooter>
          <a className={buttonVariants()} href={emailSettingsHref}>
            Open Email Settings
          </a>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
