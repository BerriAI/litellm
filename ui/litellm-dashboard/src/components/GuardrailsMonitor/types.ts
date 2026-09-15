export interface LogEntry {
  id: string;
  timestamp: string;
  action: "blocked" | "passed" | "flagged";
  model?: string;
  input_snippet?: string;
}
