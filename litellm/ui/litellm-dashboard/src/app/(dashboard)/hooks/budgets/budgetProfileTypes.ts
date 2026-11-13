export type BudgetDuration = "1mo" | "12mo";
export type CapPolicy = "pause_allow_increase" | "pause_until_reset";

export interface BudgetLimit {
  budget_duration: BudgetDuration;
  max_budget: number;
}

export interface PersonalBudget {
  budget_limits?: BudgetLimit[] | null;
  cap_policy?: CapPolicy | null;
  temp_budget_increase?: number | null;
  temp_budget_expiry?: string | null;
  temp_budget_duration?: BudgetDuration | null;
}

export interface BudgetProfileFields extends PersonalBudget {
  is_profile?: boolean;
  assignment_count?: number;
  revision?: string;
}

export interface BudgetWrite extends PersonalBudget {
  budget_id: string;
  is_profile?: boolean;
  revision?: string;
  max_budget?: number | null;
  budget_duration?: string | null;
  tpm_limit?: number | null;
  rpm_limit?: number | null;
  tpd_limit?: number | null;
}

export interface BudgetSubject {
  type: "user" | "customer_member";
  id: string;
}

export interface BudgetAssignment {
  subject_type: BudgetSubject["type"];
  subject_id: string;
  budget_id: string | null;
  budget_override: PersonalBudget | null;
  budget_override_expires_at: string | null;
  effective_budget: PersonalBudget;
  windows: {
    budget_duration: BudgetDuration;
    max_budget: number;
    reset_at: string;
    cap_policy: CapPolicy;
  }[];
  revision: string;
}

export interface BudgetAssignmentPatch {
  revision: string;
  budget_id?: string | null;
  budget_override?: PersonalBudget | null;
  budget_override_expires_at?: string | null;
}
