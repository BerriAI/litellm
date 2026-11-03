import Papa from "papaparse";
import { describe, expect, it } from "vitest";

import { TEMPLATE_CSV } from "./columns";
import { checkCsvRows, parseCsvFile, type CsvParseResult, type CsvRow } from "./parseCsv";

const HEADER = "user_email,user_role,teams,max_budget,budget_duration,models";
const KNOWN_TEAMS = ["team-a", "team-b"];

const parse = (text: string, knownTeamIds: readonly string[] = KNOWN_TEAMS): CsvParseResult =>
  checkCsvRows(Papa.parse<string[]>(text).data, knownTeamIds);

const rowsOf = (result: CsvParseResult): CsvRow[] => {
  if (result.kind !== "rows") throw new Error(`expected rows, got ${result.kind}: ${JSON.stringify(result)}`);
  return result.rows;
};

const firstRowError = (line: string, knownTeamIds?: readonly string[]): string =>
  rowsOf(parse(`${HEADER}\n${line}`, knownTeamIds))[0].error;

describe("checkCsvRows", () => {
  it("accepts the downloadable template as one valid user", () => {
    const result = parse(TEMPLATE_CSV, ["team-id-1", "team-id-2"]);

    expect(result).toMatchObject({ kind: "rows", rowError: null });
    expect(rowsOf(result)).toStrictEqual([
      {
        rowNumber: 2,
        user_email: "user@example.com",
        user_role: "internal_user",
        teams: "team-id-1,team-id-2",
        max_budget: "100",
        budget_duration: "30d",
        models: "gpt-3.5-turbo,gpt-4",
        isValid: true,
        error: "",
      },
    ]);
  });

  it("trims cells and reads columns by header name in any order", () => {
    const [row] = rowsOf(parse("models,user_role,user_email\n gpt-4o , internal_user_viewer ,  a@b.co  "));

    expect(row).toMatchObject({ user_email: "a@b.co", user_role: "internal_user_viewer", models: "gpt-4o" });
    expect(row.isValid).toBe(true);
  });

  it("skips blank lines while keeping row numbers aligned with the file", () => {
    const rows = rowsOf(parse("user_email,user_role\na@b.co,internal_user\n\nc@d.co,proxy_admin\n"));

    expect(rows.map((row) => [row.rowNumber, row.user_email])).toEqual([
      [2, "a@b.co"],
      [4, "c@d.co"],
    ]);
  });

  describe("file structure", () => {
    it("rejects an empty file", () => {
      expect(checkCsvRows([], KNOWN_TEAMS)).toStrictEqual({
        kind: "structureError",
        message: "The CSV file appears to be empty. Please upload a file with data.",
      });
    });

    it("rejects a file with only a header row", () => {
      expect(parse("user_email,user_role")).toStrictEqual({
        kind: "structureError",
        message: "The CSV file only contains headers but no user data. Please add user data to your CSV.",
      });
    });

    it("rejects a file whose first row is blank", () => {
      expect(parse("\na@b.co,internal_user")).toStrictEqual({
        kind: "structureError",
        message: "The CSV file doesn't contain any column headers. Please make sure your CSV has headers.",
      });
    });

    it("names every missing required column", () => {
      expect(parse("teams,models\nteam-a,gpt-4o")).toStrictEqual({
        kind: "structureError",
        message:
          "Your CSV is missing these required columns: user_email, user_role. Please add these columns to your CSV file.",
      });
    });

    it("names only the column that is missing", () => {
      expect(parse("user_email\na@b.co")).toMatchObject({
        message: expect.stringContaining("missing these required columns: user_role."),
      });
    });

    it("rejects a file whose data rows are all blank", () => {
      expect(parse("user_email,user_role\n\n")).toStrictEqual({
        kind: "structureError",
        message: "No valid data rows found in the CSV file. Please check your file format.",
      });
    });
  });

  it("flags a row with fewer cells than the header", () => {
    const [row] = rowsOf(parse(`${HEADER}\na@b.co,internal_user,,,`));

    const expected = {
      rowNumber: 2,
      isValid: false,
      error: "Row 2 has fewer columns than the header row. Please ensure all data is properly formatted.",
      user_email: "",
      user_role: "",
    };
    expect(row).toStrictEqual(expected);
  });

  describe("row rules", () => {
    it.each([
      ["missing email", ",internal_user,,,,", "Email is required"],
      ["email without @", "a.example.com,internal_user,,,,", "Invalid email format (must contain @ and domain)"],
      ["email without a dot", "a@localhost,internal_user,,,,", "Invalid email format (must contain @ and domain)"],
      ["missing role", "a@b.co,,,,,", "Role is required"],
      [
        "unknown role",
        "a@b.co,admin,,,,",
        'Invalid role "admin". Must be one of: proxy_admin, proxy_admin_viewer, internal_user, internal_user_viewer',
      ],
      ["non-numeric budget", "a@b.co,internal_user,,lots,,", 'Max budget "lots" must be a number'],
      ["zero budget", "a@b.co,internal_user,,0,,", "Max budget must be greater than 0"],
      ["negative budget", "a@b.co,internal_user,,-3,,", "Max budget must be greater than 0"],
      [
        "bad duration",
        "a@b.co,internal_user,,,30days,",
        'Invalid budget duration format "30days". Use format like "30d", "1mo", "2w", "6h"',
      ],
      ["unknown team", 'a@b.co,internal_user,"team-a, team-z, team-y",,,', "Unknown team(s): team-z, team-y"],
    ])("rejects a %s", (_case, line, message) => {
      expect(firstRowError(line)).toBe(message);
    });

    it.each(["proxy_admin", "proxy_admin_viewer", "internal_user", "internal_user_viewer"])(
      "accepts the %s role",
      (role) => {
        expect(firstRowError(`a@b.co,${role},,,,`)).toBe("");
      },
    );

    it.each(["30d", "12h", "15m", "2w", "1y", "1mo", "24mo"])("accepts a %s budget duration", (duration) => {
      expect(firstRowError(`a@b.co,internal_user,,,${duration},`)).toBe("");
    });

    it("accepts a decimal budget and known teams", () => {
      expect(firstRowError('a@b.co,internal_user,"team-a,team-b",0.5,,')).toBe("");
    });

    it("does not check team ids when no teams are known", () => {
      expect(firstRowError("a@b.co,internal_user,team-z,,,", [])).toBe("");
    });

    it("lists every failed rule for a row, in column order", () => {
      expect(firstRowError("nope,boss,team-z,free,forever,")).toBe(
        [
          "Invalid email format (must contain @ and domain)",
          'Invalid role "boss". Must be one of: proxy_admin, proxy_admin_viewer, internal_user, internal_user_viewer',
          'Max budget "free" must be a number',
          'Invalid budget duration format "forever". Use format like "30d", "1mo", "2w", "6h"',
          "Unknown team(s): team-z",
        ].join(", "),
      );
    });
  });

  describe("row error summary", () => {
    it("is empty when every row is valid", () => {
      expect(parse("user_email,user_role\na@b.co,internal_user\nc@d.co,proxy_admin")).toMatchObject({
        rowError: null,
      });
    });

    it("counts the invalid rows when some rows are valid", () => {
      expect(parse("user_email,user_role\na@b.co,internal_user\nbad,internal_user\nworse,nope")).toMatchObject({
        rowError: "Found 2 row(s) with errors out of 3 total rows. Please correct them before proceeding.",
      });
    });

    it("says no users are valid when every row fails", () => {
      expect(parse("user_email,user_role\nbad,internal_user")).toMatchObject({
        rowError: "No valid users found in the CSV. Please check the errors below and fix your CSV file.",
      });
    });
  });
});

describe("parseCsvFile", () => {
  it("rejects a file that is neither text/csv nor named .csv", async () => {
    const file = new File(["user_email,user_role"], "users.xlsx", { type: "application/vnd.ms-excel" });

    expect(await parseCsvFile(file, KNOWN_TEAMS)).toStrictEqual({
      kind: "fileError",
      message: "Invalid file type: users.xlsx. Please upload a CSV file (.csv extension).",
      toast: "Invalid file type. Please upload a CSV file.",
    });
  });

  it("rejects a file larger than 5MB", async () => {
    const file = new File([new Uint8Array(6 * 1024 * 1024)], "users.csv", { type: "text/csv" });

    expect(await parseCsvFile(file, KNOWN_TEAMS)).toStrictEqual({
      kind: "fileError",
      message: "File is too large (6.0 MB). Please upload a CSV file smaller than 5MB.",
      toast: null,
    });
  });
});
