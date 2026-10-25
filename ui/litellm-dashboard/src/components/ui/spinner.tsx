"use client";

import { useLanguage } from "@/contexts/LanguageContext";
import { cn } from "@/lib/cva.config";
import { Loader2Icon } from "lucide-react";

function Spinner({ className, ...props }: React.ComponentProps<"svg">) {
  const { t } = useLanguage();
  return (
    <Loader2Icon
      data-slot="spinner"
      role="status"
      aria-label={t("Loading")}
      className={cn("size-4 animate-spin", className)}
      {...props}
    />
  );
}

export { Spinner };
