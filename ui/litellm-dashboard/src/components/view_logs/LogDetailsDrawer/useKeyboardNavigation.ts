import { useShortcut } from "@/components/shared/useShortcut";

import { LogEntry } from "../columns";

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
  useShortcut("escape", onClose, { ...modal, description: "close" });
  useShortcut("j", selectNextLog, { ...modal, description: "log" });
  useShortcut("k", selectPreviousLog, { ...modal, description: "log" });
  return { selectNextLog, selectPreviousLog };
}
