import type { UserSearchFields } from "./schema";

export const emptyUserSearch = (defaultRole: string): UserSearchFields => ({
  user_email: undefined,
  user_id: undefined,
  role: defaultRole,
});
