export interface User {
  user_id: string;
  user_email: string;
  role?: string;
}

export interface UserOption {
  label: string;
  value: string;
  user: User;
}

export type Model = { id: string };

export type ModelAvailableResponse = { data: Model[] };

export type ModelAliases = { [key: string]: string };
