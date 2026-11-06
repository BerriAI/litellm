import { useTranslation } from "react-i18next";

import { useShortcut } from "@/components/shared/useShortcut";

import { LogEntry } from "../types";

interface UseKeyboardNavigationProps {
  isOpen: boolean;
  currentLog: LogEntry | null;
  allLogs: LogEntry[];
  onClose: () => void;
  onSelectLog?: (log: LogEntry) => void;
}

/** J and K step through the open log's neighbours and Escape closes the drawer. */
export function useKeyboardNavigation({
  isOpen,
  currentLog,
  allLogs,
  onClose,
  onSelectLog,
}: UseKeyboardNavigationProps) {
  const { t } = useTranslation();
  const index = currentLog ? allLogs.findIndex((log) => log.request_id === currentLog.request_id) : -1;
  const selectAt = (next: number) => {
    const log = allLogs[next];
    if (currentLog && log) onSelectLog?.(log);
  };
  const selectNextLog = () => selectAt(index + 1);
  const selectPreviousLog = () => {
    if (index > 0) selectAt(index - 1);
  };
  const modal = { layer: "modal", enabled: isOpen } as const;
  useShortcut("escape", onClose, { ...modal, description: t("logs.detail.drawer.shortcuts.close") });
  useShortcut("j", selectNextLog, { ...modal, description: t("logs.detail.drawer.shortcuts.log") });
  useShortcut("k", selectPreviousLog, { ...modal, description: t("logs.detail.drawer.shortcuts.log") });
  return { selectNextLog, selectPreviousLog };
}
