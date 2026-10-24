import Papa from "papaparse";

import { USER_ROLES } from "../userRoles";

export const TEMPLATE_COLUMNS = [
  {
    name: "user_email",
    required: true,
    help: "User's email address (required)",
    example: "user@example.com",
  },
  {
    name: "user_role",
    required: true,
    help: `User's role (one of: ${USER_ROLES.map((role) => `"${role}"`).join(", ")})`,
    example: "internal_user",
  },
  {
    name: "teams",
    required: false,
    help: 'Comma-separated team IDs (e.g., "team-1,team-2")',
    example: "team-id-1,team-id-2",
  },
  {
    name: "max_budget",
    required: false,
    help: 'Maximum budget as a number (e.g., "100")',
    example: "100",
  },
  {
    name: "budget_duration",
    required: false,
    help: 'Budget reset period (e.g., "30d", "1mo")',
    example: "30d",
  },
  {
    name: "models",
    required: false,
    help: 'Comma-separated allowed models (e.g., "gpt-3.5-turbo,gpt-4")',
    example: "gpt-3.5-turbo,gpt-4",
  },
] as const;

export type ColumnName = (typeof TEMPLATE_COLUMNS)[number]["name"];

export const REQUIRED_COLUMNS: readonly ColumnName[] = TEMPLATE_COLUMNS.filter((column) => column.required).map(
  (column) => column.name,
);

export const TEMPLATE_CSV = Papa.unparse([
  TEMPLATE_COLUMNS.map((column) => column.name),
  TEMPLATE_COLUMNS.map((column) => column.example),
]);
