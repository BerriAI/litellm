import Papa from "papaparse";

import { TeamListCallOptions, TeamsResponse, teamListCall } from "@/app/(dashboard)/hooks/teams/useTeams";

import { Team } from "../key_team_helpers/key_list";
import { apiClient } from "../networking";

export interface TeamMemberBudget {
  budget_id: string;
  max_budget?: number | null;
  budget_duration?: string | null;
  tpm_limit?: number | null;
  rpm_limit?: number | null;
}

export const TEAMS_EXPORT_PAGE_SIZE = 100;

type FetchTeamsPage = (page: number, pageSize: number) => Promise<TeamsResponse>;

export const fetchAllTeams = async (fetchPage: FetchTeamsPage): Promise<Team[]> => {
  const firstPage = await fetchPage(1, TEAMS_EXPORT_PAGE_SIZE);
  const totalPages = firstPage.total_pages ?? 1;
  if (totalPages <= 1) return firstPage.teams;

  const remainingPages = await Promise.all(
    Array.from({ length: totalPages - 1 }, (_, i) => fetchPage(i + 2, TEAMS_EXPORT_PAGE_SIZE)),
  );
  return [firstPage, ...remainingPages].flatMap((page) => page.teams);
};

const teamMemberBudgetId = (team: Team): string | null => {
  const id = team.metadata?.team_member_budget_id;
  return typeof id === "string" && id.length > 0 ? id : null;
};

export const collectTeamMemberBudgetIds = (teams: Team[]): string[] =>
  Array.from(new Set(teams.map(teamMemberBudgetId).filter((id): id is string => id !== null)));

const cell = (value: string | number | boolean | null | undefined): string | number | boolean => value ?? "";

export const buildTeamsCsvRows = (
  teams: Team[],
  budgets: TeamMemberBudget[],
): Record<string, string | number | boolean>[] => {
  const budgetsById = new Map(budgets.map((budget) => [budget.budget_id, budget]));
  return teams.map((team) => {
    const budgetId = teamMemberBudgetId(team);
    const memberBudget = budgetId ? budgetsById.get(budgetId) : undefined;
    return {
      "Team Alias": cell(team.team_alias),
      "Team ID": cell(team.team_id),
      "Organization ID": cell(team.organization_id),
      Models: (team.models ?? []).join(", "),
      "Max Budget (USD)": cell(team.max_budget),
      "Budget Duration": cell(team.budget_duration),
      "Budget Reset At": cell(team.budget_reset_at),
      "Spend (USD)": cell(team.spend),
      "TPM Limit": cell(team.tpm_limit),
      "RPM Limit": cell(team.rpm_limit),
      "Team Member Budget (USD)": cell(memberBudget?.max_budget),
      "Team Member Budget Duration": cell(memberBudget?.budget_duration),
      "Team Member TPM Limit": cell(memberBudget?.tpm_limit),
      "Team Member RPM Limit": cell(memberBudget?.rpm_limit),
      Members: cell(team.members_count ?? team.members_with_roles?.length),
      Keys: cell(team.keys_count ?? team.keys?.length),
      Blocked: cell(team.blocked),
      "Created At": cell(team.created_at),
    };
  });
};

export const buildTeamsCsv = (teams: Team[], budgets: TeamMemberBudget[]): string =>
  Papa.unparse(buildTeamsCsvRows(teams, budgets), { escapeFormulae: true });

const downloadCsv = (csv: string, fileName: string): void => {
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

export const exportTeamsToCsv = async (accessToken: string, options: TeamListCallOptions): Promise<number> => {
  const teams = await fetchAllTeams((page, pageSize) => teamListCall(accessToken, page, pageSize, options));
  const budgetIds = collectTeamMemberBudgetIds(teams);
  const budgets = budgetIds.length
    ? await apiClient.post<TeamMemberBudget[]>("/budget/info", { accessToken, body: { budgets: budgetIds } })
    : [];
  downloadCsv(buildTeamsCsv(teams, budgets), `teams_export_${new Date().toISOString().split("T")[0]}.csv`);
  return teams.length;
};
