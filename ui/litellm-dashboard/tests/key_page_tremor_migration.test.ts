import { describe, expect, it } from "vitest";
import { readFileSync } from "fs";
import path from "path";

const dashboardRoot = path.resolve(__dirname, "..");

const keyPageFiles = [
  "src/components/user_dashboard.tsx",
  "src/components/key_value_input.tsx",
  "src/components/templates/key_info_view.tsx",
  "src/components/templates/key_edit_view.tsx",
  "src/components/organisms/create_key_button.tsx",
];

const tremorTablePrimitives = new Set([
  "Table",
  "TableBody",
  "TableCell",
  "TableHead",
  "TableHeaderCell",
  "TableRow",
]);

const getTremorImports = (relativePath: string): string[] => {
  const contents = readFileSync(path.join(dashboardRoot, relativePath), "utf8");
  const tremorImportRegex = /^import\s*\{([^}]*)\}\s*from\s*["']@tremor\/react["'];?/gm;
  const imports: string[] = [];
  let match: RegExpExecArray | null;

  while ((match = tremorImportRegex.exec(contents)) !== null) {
    imports.push(
      ...match[1]
        .split(",")
        .map((name) => name.trim().split(/\s+as\s+/)[0])
        .filter(Boolean),
    );
  }

  return imports;
};

describe("key page Tremor migration", () => {
  it("should not import Tremor components in key page create, edit, detail, or layout files", () => {
    const importsByFile = keyPageFiles
      .map((relativePath) => [relativePath, getTremorImports(relativePath)] as const)
      .filter(([, imports]) => imports.length > 0);

    expect(importsByFile).toEqual([]);
  });

  it("should only keep Tremor table primitives in the virtual keys table", () => {
    const imports = getTremorImports("src/components/VirtualKeysPage/VirtualKeysTable.tsx");
    const nonTableImports = imports.filter((name) => !tremorTablePrimitives.has(name));

    expect(nonTableImports).toEqual([]);
  });
});
