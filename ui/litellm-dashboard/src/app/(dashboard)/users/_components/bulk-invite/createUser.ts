import { getUiPageUrl } from "@/components/networking";

import type { CsvRow } from "./parseCsv";

export interface CreateUserRequest {
  user_email: string;
  user_role: string;
  teams?: string[];
  models?: string[];
  max_budget?: number;
  budget_duration?: string;
}

export interface CreatedUser {
  key?: string | null;
  user_id?: string | null;
  data?: { user_id?: string | null } | null;
  error?: string;
}

export interface BulkInviteApi {
  createUser: (request: CreateUserRequest) => Promise<CreatedUser | null>;
  createInvitation: (userId: string) => Promise<{ id: string }>;
}

export interface InviteSettings {
  ssoEnabled: boolean;
  baseUrl: string;
}

export type CreateUserOutcome =
  | { status: "success"; key: string; invitation_link: string }
  | { status: "success"; key: string; error: string }
  | { status: "failed"; error: string };

const splitList = (value: string | undefined): string[] =>
  (value ?? "")
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);

const buildRequest = (row: CsvRow): CreateUserRequest => {
  const teams = splitList(row.teams);
  const models = splitList(row.models);
  const maxBudget = parseFloat(row.max_budget ?? "");
  const budgetDuration = row.budget_duration?.trim();
  return {
    user_email: row.user_email,
    user_role: row.user_role,
    ...(teams.length > 0 && { teams }),
    ...(models.length > 0 && { models }),
    ...(maxBudget > 0 && { max_budget: maxBudget }),
    ...(budgetDuration && { budget_duration: budgetDuration }),
  };
};

const invitationLink = async (
  userId: string | null | undefined,
  api: BulkInviteApi,
  settings: InviteSettings,
): Promise<string> => {
  if (settings.ssoEnabled) return new URL("/ui", settings.baseUrl).toString();
  if (!userId) throw new Error("Created user has no user_id");
  const invitation = await api.createInvitation(userId);
  return getUiPageUrl(new URL(`/ui/onboarding?invitation_id=${invitation.id}`, settings.baseUrl).toString());
};

const withInvitation = async (
  created: CreatedUser,
  key: string,
  api: BulkInviteApi,
  settings: InviteSettings,
): Promise<CreateUserOutcome> => {
  try {
    const link = await invitationLink(created.data?.user_id || created.user_id, api, settings);
    return { status: "success", key, invitation_link: link };
  } catch (inviteError) {
    console.error("Error creating invitation:", inviteError);
    return { status: "success", key, error: "User created but failed to generate invitation link" };
  }
};

const failureMessage = (error: unknown): string => (error instanceof Error && error.message) || String(error);

export const createUserFromRow = async (
  row: CsvRow,
  api: BulkInviteApi,
  settings: InviteSettings,
): Promise<CreateUserOutcome> => {
  try {
    const created = await api.createUser(buildRequest(row));
    const key = created?.key || created?.user_id;
    if (!created || !key) return { status: "failed", error: created?.error || "Failed to create user" };
    return await withInvitation(created, key, api, settings);
  } catch (error) {
    console.error("Caught error:", error);
    return { status: "failed", error: failureMessage(error) };
  }
};
