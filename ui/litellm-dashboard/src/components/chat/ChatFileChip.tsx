import { useLanguage } from "@/contexts/LanguageContext";
import { FileText } from "lucide-react";
import type { ChatFileAttachment } from "./types";

export default function ChatFileChip({ file }: { file: ChatFileAttachment }) {
  const { t } = useLanguage();
  return (
    <div className="flex min-w-0 items-center gap-2 rounded-lg border bg-muted px-3 py-2 text-sm">
      <FileText className="size-5 shrink-0 text-muted-foreground" aria-hidden="true" />
      <div className="min-w-0">
        <p className="truncate" title={file.name}>
          {file.name}
        </p>
        <p className="text-xs text-muted-foreground">
          {file.url ? t("Attached file") : t("File unavailable — start a new chat and attach it again")}
        </p>
      </div>
    </div>
  );
}
