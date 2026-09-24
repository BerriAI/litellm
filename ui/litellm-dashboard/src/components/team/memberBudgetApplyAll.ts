export const MEMBER_BUDGET_BULK_CHUNK_SIZE = 500;

interface MembershipBudgetRow {
  user_id: string | null;
  budget_source: string;
}

export const customBudgetMemberUserIds = (memberships: readonly MembershipBudgetRow[]): string[] =>
  memberships.filter((m) => m.budget_source === "custom" && m.user_id).map((m) => m.user_id as string);

export const shouldPromptMemberBudgetApplyAll = (
  nextBudget: number | undefined,
  previousBudget: number | null | undefined,
  customBudgetUserIds: string[],
): boolean => {
  const budgetChanged = typeof nextBudget === "number" && nextBudget > 0 && nextBudget !== previousBudget;
  return budgetChanged && customBudgetUserIds.length > 0;
};

export const chunkMemberIds = (userIds: string[], size = MEMBER_BUDGET_BULK_CHUNK_SIZE): string[][] => {
  const chunks: string[][] = [];
  for (let i = 0; i < userIds.length; i += size) {
    chunks.push(userIds.slice(i, i + size));
  }
  return chunks;
};
