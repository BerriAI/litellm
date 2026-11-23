"use client";

import { useState } from "react";
import { useTranslation } from "react-i18next";
import { Check, Copy, Play, Plus } from "lucide-react";
import { Button } from "@/components/ui/button";
import { uiHref } from "@/utils/uiHref";
import type { Recommendation } from "./api";

export function ModelActions({ model }: { model: Recommendation }) {
  const { t } = useTranslation(undefined, { keyPrefix: "modelRecommender" });
  const [copied, setCopied] = useState(false);
  const [copyError, setCopyError] = useState(false);
  const onGateway = model.on_gateway !== false;
  const destination = onGateway
    ? `playground?model=${encodeURIComponent(model.model_group)}`
    : `models-and-endpoints?tab=add&provider=${encodeURIComponent(model.providers[0] ?? "")}&litellm_model=${encodeURIComponent(model.model_group)}`;
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(model.model_group);
      setCopied(true);
      setCopyError(false);
    } catch {
      setCopyError(true);
    }
  };
  return (
    <div className="mt-auto space-y-2">
      <div className="flex flex-wrap gap-2">
        <Button size="sm" nativeButton={false} role="link" render={<a href={uiHref(destination)} />}>
          {onGateway ? <Play /> : <Plus />}
          {onGateway ? t("playground") : t("addModel")}
        </Button>
        <Button size="sm" variant="outline" onClick={copy}>
          {copied ? <Check /> : <Copy />}
          {copied ? t("copied") : t("copy")}
        </Button>
      </div>
      {copyError && (
        <p role="alert" className="text-xs text-destructive">
          {t("copyFailed")}
        </p>
      )}
    </div>
  );
}
