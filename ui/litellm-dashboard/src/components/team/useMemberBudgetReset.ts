import { useCallback, useRef, useState } from "react";
import { toast } from "@/lib/toast";
import {
  memberBudgetUpdateMessage,
  type MemberBudgetResetPending,
  type MemberBudgetUpdateMode,
  type TeamUpdatePayload,
} from "./memberBudgetReset";

export type MemberBudgetResetState =
  | { phase: "idle" }
  | { phase: "prompting" | "saving"; pending: MemberBudgetResetPending };

export interface MemberBudgetResetGateway {
  saveTeam: (updateData: TeamUpdatePayload) => Promise<{ member_budgets_updated?: number }>;
  refreshTeamData: () => Promise<void>;
}

export const useMemberBudgetReset = (gateway: MemberBudgetResetGateway) => {
  const [state, setState] = useState<MemberBudgetResetState>({ phase: "idle" });
  const activeRun = useRef<object | null>(null);

  const prompt = (pending: MemberBudgetResetPending) => setState({ phase: "prompting", pending });

  const save = async (mode: MemberBudgetUpdateMode) => {
    if (state.phase !== "prompting" || activeRun.current !== null) return;
    const { pending } = state;
    const run = {};
    activeRun.current = run;
    setState({ phase: "saving", pending });
    try {
      const result = await gateway.saveTeam({ ...pending.updateData, team_member_budget_update_mode: mode });
      if (activeRun.current !== run) return;
      setState({ phase: "idle" });
      toast.success(
        mode === "keep"
          ? "Team settings updated successfully"
          : memberBudgetUpdateMessage(result.member_budgets_updated ?? 0),
      );
      await gateway.refreshTeamData();
    } catch {
      if (activeRun.current === run) setState({ phase: "prompting", pending });
    } finally {
      if (activeRun.current === run) activeRun.current = null;
    }
  };

  const dismiss = useCallback(() => {
    activeRun.current = null;
    setState({ phase: "idle" });
  }, []);

  return { state, prompt, save, dismiss };
};
