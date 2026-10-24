import Papa from "papaparse";

import { REQUIRED_COLUMNS, type ColumnName } from "./columns";
import { csvRowSchema } from "./schema";

const MAX_FILE_BYTES = 5 * 1024 * 1024;

export interface CsvRow {
  rowNumber: number;
  user_email: string;
  user_role: string;
  teams?: string;
  max_budget?: string;
  budget_duration?: string;
  models?: string;
  isValid: boolean;
  error: string;
}

export type CsvParseResult =
  | { kind: "fileError"; message: string; toast: string | null }
  | { kind: "structureError"; message: string }
  | { kind: "parseError"; message: string }
  | { kind: "rows"; rows: CsvRow[]; rowError: string | null };

type RowSchema = ReturnType<typeof csvRowSchema>;

const isEmptyRow = (cells: string[]): boolean => cells.length === 0 || (cells.length === 1 && cells[0] === "");

const checkRow = (headers: string[], cells: string[], rowNumber: number, schema: RowSchema): CsvRow => {
  if (cells.length < headers.length) {
    return {
      rowNumber,
      isValid: false,
      error: `Row ${rowNumber} has fewer columns than the header row. Please ensure all data is properly formatted.`,
      user_email: "",
      user_role: "",
    };
  }

  const cell = (name: ColumnName): string | undefined => cells[headers.indexOf(name)]?.trim();
  const values = {
    user_email: cell("user_email") || "",
    user_role: cell("user_role") || "",
    teams: cell("teams"),
    max_budget: cell("max_budget"),
    budget_duration: cell("budget_duration"),
    models: cell("models"),
  };
  const result = schema.safeParse(values);
  const error = result.success ? "" : result.error.issues.map((issue) => issue.message).join(", ");
  return { ...values, rowNumber, isValid: result.success, error };
};

const rowErrorSummary = (rows: CsvRow[]): string | null => {
  const invalidCount = rows.filter((row) => !row.isValid).length;
  if (invalidCount === 0) return null;
  if (invalidCount === rows.length) {
    return "No valid users found in the CSV. Please check the errors below and fix your CSV file.";
  }
  return `Found ${invalidCount} row(s) with errors out of ${rows.length} total rows. Please correct them before proceeding.`;
};

const checkHeaders = (headers: string[]): string | null => {
  if (headers.length === 0 || (headers.length === 1 && headers[0] === "")) {
    return "The CSV file doesn't contain any column headers. Please make sure your CSV has headers.";
  }
  const missingColumns = REQUIRED_COLUMNS.filter((column) => !headers.includes(column));
  if (missingColumns.length > 0) {
    return `Your CSV is missing these required columns: ${missingColumns.join(", ")}. Please add these columns to your CSV file.`;
  }
  return null;
};

export const checkCsvRows = (data: string[][], knownTeamIds: readonly string[]): CsvParseResult => {
  if (data.length === 0) {
    return { kind: "structureError", message: "The CSV file appears to be empty. Please upload a file with data." };
  }
  if (data.length === 1) {
    return {
      kind: "structureError",
      message: "The CSV file only contains headers but no user data. Please add user data to your CSV.",
    };
  }

  const [headers, ...body] = data;
  const headerError = checkHeaders(headers);
  if (headerError) return { kind: "structureError", message: headerError };

  const schema = csvRowSchema(knownTeamIds);
  const rows = body.flatMap((cells, index) => (isEmptyRow(cells) ? [] : [checkRow(headers, cells, index + 2, schema)]));
  if (rows.length === 0) {
    return {
      kind: "structureError",
      message: "No valid data rows found in the CSV file. Please check your file format.",
    };
  }
  return { kind: "rows", rows, rowError: rowErrorSummary(rows) };
};

const checkFile = (file: File): CsvParseResult | null => {
  if (file.type !== "text/csv" && !file.name.endsWith(".csv")) {
    return {
      kind: "fileError",
      message: `Invalid file type: ${file.name}. Please upload a CSV file (.csv extension).`,
      toast: "Invalid file type. Please upload a CSV file.",
    };
  }
  if (file.size > MAX_FILE_BYTES) {
    return {
      kind: "fileError",
      message: `File is too large (${(file.size / (1024 * 1024)).toFixed(1)} MB). Please upload a CSV file smaller than 5MB.`,
      toast: null,
    };
  }
  return null;
};

export const parseCsvFile = (file: File, knownTeamIds: readonly string[]): Promise<CsvParseResult> => {
  const fileProblem = checkFile(file);
  if (fileProblem) return Promise.resolve(fileProblem);

  return new Promise((resolve) => {
    Papa.parse<string[]>(file, {
      complete: (results) => resolve(checkCsvRows(results.data, knownTeamIds)),
      error: (error) => resolve({ kind: "parseError", message: `Failed to parse CSV file: ${error.message}` }),
      header: false,
    });
  });
};
