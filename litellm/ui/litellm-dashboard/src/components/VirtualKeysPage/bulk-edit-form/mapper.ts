import type { BulkKeyEditAction, BulkKeyEditValues } from "../VirtualKeysBulkEditDialog";

export const mapBulkKeyEdit = (
  action: BulkKeyEditAction,
  { teamId, userId, duration }: { teamId: string | null; userId: string | null; duration: string | null },
  budget: number | null,
): BulkKeyEditValues | null => {
  switch (action) {
    case "team":
      return teamId ? { team_id: teamId } : null;
    case "user":
      return userId ? { user_id: userId } : null;
    case "budget":
      return budget !== null && duration ? { max_budget: budget, budget_duration: duration } : null;
  }
};
