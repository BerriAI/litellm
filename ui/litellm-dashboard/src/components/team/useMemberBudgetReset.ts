import { useCallback, useRef, useState } from "react";
import { toast } from "@/lib/toast";
import { formatNumberWithCommas } from "@/utils/dataUtils";
import {
  pluralize,
  type MemberBudgetBackfill,
  type MemberBudgetBackfillResult,
  type MemberBudgetResetPending,
  type TeamUpdatePayload,
} from "./memberBudgetReset";

export type MemberBudgetResetState =
  | { phase: "idle" }
  | { phase: "prompting"; pending: MemberBudgetResetPending }
  | { phase: "saving"; pending: MemberBudgetResetPending };

export interface MemberBudgetResetGateway {
  saveTeam: (updateData: TeamUpdatePayload) => Promise<MemberBudgetBackfillResult | undefined>;
  refreshTeamData: () => Promise<void>;
}

export const backfillToastMessage = (result: MemberBudgetBackfillResult | undefined, newBudget: number): string => {
  if (result === undefined) return "Team settings updated successfully";
  const moved = `Moved ${result.moved_to_default} ${pluralize(result.moved_to_default, "member", "members")} to the $${formatNumberWithCommas(newBudget, 2)} default`;
  const skipped = result.skipped_other_reset_period;
  if (skipped === 0) return moved;
  return `${moved}. ${skipped} ${pluralize(skipped, "member resets", "members reset")} on a different schedule and ${pluralize(skipped, "was", "were")} not changed`;
};

export const useMemberBudgetReset = (gateway: MemberBudgetResetGateway) => {
  const [state, setState] = useState<MemberBudgetResetState>({ phase: "idle" });
  const activeRun = useRef<object | null>(null);

  const prompt = (pending: MemberBudgetResetPending) => setState({ phase: "prompting", pending });

  const save = async (backfill: MemberBudgetBackfill) => {
    if (state.phase !== "prompting") return;
    const { pending } = state;
    const run = {};
    activeRun.current = run;
    setState({ phase: "saving", pending });
    try {
      const result = await gateway.saveTeam({ ...pending.updateData, team_member_budget_backfill: backfill });
      if (activeRun.current !== run) return;
      toast.success(backfillToastMessage(backfill === "none" ? undefined : result, pending.newBudget));
      setState({ phase: "idle" });
      await gateway.refreshTeamData();
    } catch (error) {
      console.error("Error updating team:", error);
      if (activeRun.current === run) setState({ phase: "prompting", pending });
    }
  };

  const dismiss = useCallback(() => {
    activeRun.current = null;
    setState({ phase: "idle" });
  }, []);

  return { state, prompt, save, dismiss };
};
