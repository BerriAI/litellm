import Papa from "papaparse";

import type { TeamUserSpendResponse } from "@/components/networking";

export type TeamUserSpendRow = TeamUserSpendResponse["results"][number];

export const NO_USER_LABEL = "(no user)";

export const userLabel = (row: TeamUserSpendRow): string => {
  const identity = row.user_email || row.user_alias;
  return identity || row.user_id || NO_USER_LABEL;
};

export const teamLabel = (row: TeamUserSpendRow): string => row.team_alias || row.team_id;

export const teamUserSpendRowId = (row: TeamUserSpendRow): string => `${row.team_id}\u0000${row.user_id}`;

export const sortBySpendDesc = (rows: readonly TeamUserSpendRow[]): TeamUserSpendRow[] =>
  [...rows].sort((a, b) => b.spend - a.spend || teamLabel(a).localeCompare(teamLabel(b)));

export const buildTeamUserSpendCsv = (response: TeamUserSpendResponse): string =>
  Papa.unparse(
    sortBySpendDesc(response.results).map((row) => ({
      "Start Date": response.start_date,
      "End Date": response.end_date,
      Team: teamLabel(row),
      "Team ID": row.team_id,
      User: userLabel(row),
      "User ID": row.user_id,
      "User Email": row.user_email ?? "",
      "Spend (USD)": row.spend,
      Requests: row.api_requests,
      Successful: row.successful_requests,
      Failed: row.failed_requests,
      "Prompt Tokens": row.prompt_tokens,
      "Completion Tokens": row.completion_tokens,
      "Total Tokens": row.total_tokens,
    })),
    { escapeFormulae: true },
  );

export const teamUserSpendCsvFileName = (response: TeamUserSpendResponse): string =>
  `team_user_spend_${response.start_date}_to_${response.end_date}.csv`;

export const downloadCsv = (csv: string, fileName: string): void => {
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = fileName;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  window.URL.revokeObjectURL(url);
};
