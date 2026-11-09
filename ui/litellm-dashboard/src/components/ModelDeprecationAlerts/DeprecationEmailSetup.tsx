"use client";

import type { ParseKeys } from "i18next";
import { Mail } from "lucide-react";
import { useTranslation } from "react-i18next";
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
const setupSteps: readonly { titleKey: ParseKeys; descriptionKey: ParseKeys }[] = [
  {
    titleKey: "alerts.emailSetup.steps.connect.title",
    descriptionKey: "alerts.emailSetup.steps.connect.description",
  },
  {
    titleKey: "alerts.emailSetup.steps.enable.title",
    descriptionKey: "alerts.emailSetup.steps.enable.description",
  },
  {
    titleKey: "alerts.emailSetup.steps.reminders.title",
    descriptionKey: "alerts.emailSetup.steps.reminders.description",
  },
];

export default function DeprecationEmailSetup() {
  const { t } = useTranslation();
  return (
    <Dialog>
      <div className="flex flex-wrap items-center justify-between gap-2 border-b pb-3 text-xs text-muted-foreground">
        <span className="flex items-center gap-2">
          <Mail className="size-3.5 shrink-0" aria-hidden />
          {t("alerts.emailSetup.summary")}
        </span>
        <DialogTrigger render={<Button variant="link" size="sm" />}>{t("alerts.emailSetup.trigger")}</DialogTrigger>
      </div>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>{t("alerts.emailSetup.title")}</DialogTitle>
          <DialogDescription>{t("alerts.emailSetup.description")}</DialogDescription>
        </DialogHeader>
        <ol className="my-3 space-y-5">
          {setupSteps.map((step, index) => (
            <li key={step.titleKey} className="flex gap-3">
              <span
                className="flex size-7 shrink-0 items-center justify-center rounded-full border text-xs font-medium"
                aria-hidden
              >
                {index + 1}
              </span>
              <div>
                <h3 className="text-sm font-medium">{t(step.titleKey)}</h3>
                <p className="mt-1 text-sm leading-relaxed text-muted-foreground">{t(step.descriptionKey)}</p>
              </div>
            </li>
          ))}
        </ol>
        <DialogFooter>
          <a className={buttonVariants()} href={emailSettingsHref}>
            {t("alerts.emailSetup.openSettings")}
          </a>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
