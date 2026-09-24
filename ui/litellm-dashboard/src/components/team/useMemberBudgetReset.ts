import { useCallback, useState } from "react";
import type { components } from "@/lib/http/schema";
import { toast } from "@/lib/toast";
import {
  chunk,
  MAX_BULK_TEAM_MEMBER_BUDGET_UPDATES,
  pluralize,
  type MemberBudgetResetPending,
} from "./memberBudgetReset";

export type MemberBudgetBulkResult = components["schemas"]["TeamMemberBudgetUpdateResult"];

export type MemberBudgetResetState =
  | { phase: "idle" }
  | { phase: "prompting"; pending: MemberBudgetResetPending }
  | { phase: "resetting"; pending: MemberBudgetResetPending; attempted: number }
  | { phase: "resetFailed"; pending: MemberBudgetResetPending; attempted: number };

export interface MemberBudgetResetGateway {
  saveTeam: (updateData: Record<string, unknown>) => Promise<void>;
  resetMemberBudgets: (teamId: string, userIds: readonly string[]) => Promise<MemberBudgetBulkResult[]>;
  refreshTeamData: () => Promise<void>;
}

export const useMemberBudgetReset = (gateway: MemberBudgetResetGateway) => {
  const [state, setState] = useState<MemberBudgetResetState>({ phase: "idle" });

  const runReset = async (pending: MemberBudgetResetPending, attempted: number) => {
    const { resetMemberBudgets, refreshTeamData } = gateway;
    const results: MemberBudgetBulkResult[] = [];
    try {
      for (const ids of chunk(pending.userIds.slice(attempted), MAX_BULK_TEAM_MEMBER_BUDGET_UPDATES)) {
        results.push(...(await resetMemberBudgets(pending.teamId, ids)));
        attempted += ids.length;
      }
    } catch (error) {
      console.error("Error resetting member budgets:", error);
      const total = pending.userIds.length;
      if (attempted > 0) {
        toast.error(
          `Reset ${attempted} of ${total} member ${pluralize(total, "budget", "budgets")}; the rest could not be reset`,
        );
      } else {
        toast.fromError("Team updated, but member budgets could not be reset");
      }
      setState({ phase: "resetFailed", pending, attempted });
      await refreshTeamData();
      return;
    }
    const failed = results.filter((r) => !r.success);
    if (failed.length > 0) {
      toast.error(
        `Team updated, but ${failed.length} member ${pluralize(failed.length, "budget", "budgets")} could not be reset`,
      );
    } else {
      toast.success(`Reset ${attempted} member ${pluralize(attempted, "budget", "budgets")} to the team default`);
    }
    setState({ phase: "idle" });
    await refreshTeamData();
  };

  const prompt = (pending: MemberBudgetResetPending) => setState({ phase: "prompting", pending });

  const reset = async () => {
    if (state.phase !== "prompting") return;
    const { pending } = state;
    setState({ phase: "resetting", pending, attempted: 0 });
    try {
      await gateway.saveTeam(pending.updateData);
    } catch (error) {
      console.error("Error updating team:", error);
      setState({ phase: "prompting", pending });
      return;
    }
    await runReset(pending, 0);
  };

  const retry = async () => {
    if (state.phase !== "resetFailed") return;
    const { pending, attempted } = state;
    setState({ phase: "resetting", pending, attempted });
    await runReset(pending, attempted);
  };

  const keepCustom = async () => {
    if (state.phase !== "prompting") return;
    const { pending } = state;
    setState({ phase: "idle" });
    try {
      await gateway.saveTeam(pending.updateData);
      toast.success("Team settings updated successfully");
    } catch (error) {
      console.error("Error updating team:", error);
    }
    await gateway.refreshTeamData();
  };

  const dismiss = useCallback(() => setState({ phase: "idle" }), []);

  return { state, prompt, reset, retry, keepCustom, dismiss };
};
