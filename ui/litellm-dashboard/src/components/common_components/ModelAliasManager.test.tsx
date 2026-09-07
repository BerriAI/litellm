import { readFileSync } from "fs";
import { resolve } from "path";

import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import ModelAliasManager from "./ModelAliasManager";

vi.mock("./ModelSelector", () => ({
  default: ({ value, onChange }: { value?: string; onChange?: (v: string) => void }) => (
    <input aria-label="target model" value={value ?? ""} onChange={(event) => onChange?.(event.target.value)} />
  ),
}));

vi.mock("@/lib/toast", () => ({
  toast: { success: vi.fn(), fromError: vi.fn(), info: vi.fn() },
}));

const SOURCE_PATH = resolve(process.cwd(), "src/components/common_components/ModelAliasManager.tsx");
const HARDCODED_PALETTE =
  /\b(?:text|bg|border|hover:bg|hover:text|dark:bg|dark:text)-(?:gray|slate|zinc|neutral|stone|red|blue|green|yellow|amber|orange)-\d+\b/g;
const SEMANTIC_TOKEN =
  /\b(?:text|bg|border)-(?:foreground|muted-foreground|muted|background|card|primary|secondary|destructive|border|input|accent)\b/g;

const NO_ALIASES: Record<string, string> = {};

const renderInForm = (onSubmit = vi.fn(), onAliasUpdate = vi.fn()) => {
  render(
    <form
      onSubmit={(event) => {
        event.preventDefault();
        onSubmit();
      }}
    >
      <ModelAliasManager
        accessToken="tok"
        initialModelAliases={NO_ALIASES}
        onAliasUpdate={onAliasUpdate}
        showExampleConfig={false}
      />
      <button type="submit">Save team</button>
    </form>,
  );
  return { onSubmit, onAliasUpdate, user: userEvent.setup() };
};

const addAlias = async (user: ReturnType<typeof userEvent.setup>, name: string, model: string) => {
  await user.type(screen.getByLabelText("Alias Name"), name);
  await user.type(screen.getByLabelText("target model"), model);
  await user.click(screen.getByRole("button", { name: /add alias/i }));
};

describe("ModelAliasManager dark-mode tokens", () => {
  const source = readFileSync(SOURCE_PATH, "utf8");

  it("reads the component source it is asserting about", () => {
    expect(source.length).toBeGreaterThan(0);
    expect(source).toContain("Manage Existing Aliases");
  });

  it("uses semantic colour tokens and no hardcoded palette classes", () => {
    expect(source.match(SEMANTIC_TOKEN) ?? []).not.toHaveLength(0);
    expect(source.match(HARDCODED_PALETTE) ?? []).toHaveLength(0);
  });
});

describe("ModelAliasManager", () => {
  it("reports a typed alias and target model to the parent", async () => {
    const { onAliasUpdate, user } = renderInForm();

    await addAlias(user, "fast-model", "gpt-4o");

    expect(onAliasUpdate).toHaveBeenCalledWith({ "fast-model": "gpt-4o" });
    expect(screen.getByText("fast-model")).toBeInTheDocument();
  });

  it("keeps Add Alias disabled until both fields have a value", async () => {
    const { onAliasUpdate, user } = renderInForm();
    const addButton = screen.getByRole("button", { name: /add alias/i });

    expect(addButton).toBeDisabled();
    await user.type(screen.getByLabelText("Alias Name"), "fast-model");
    expect(addButton).toBeDisabled();

    await user.type(screen.getByLabelText("target model"), "gpt-4o");
    expect(addButton).toBeEnabled();
    await user.click(addButton);
    expect(onAliasUpdate).toHaveBeenCalledWith({ "fast-model": "gpt-4o" });
  });

  it("renames an alias through the row editor", async () => {
    const { onAliasUpdate, user } = renderInForm();
    await addAlias(user, "fast-model", "gpt-4o");

    await user.click(screen.getByRole("button", { name: "Edit fast-model" }));
    const rowInput = screen.getByLabelText("Edit alias name");
    await user.clear(rowInput);
    await user.type(rowInput, "renamed-model");
    await user.click(screen.getByRole("button", { name: "Save" }));

    expect(onAliasUpdate).toHaveBeenLastCalledWith({ "renamed-model": "gpt-4o" });
  });

  it("drops an alias through the row delete action", async () => {
    const { onAliasUpdate, user } = renderInForm();
    await addAlias(user, "fast-model", "gpt-4o");

    await user.click(screen.getByRole("button", { name: "Delete fast-model" }));

    expect(onAliasUpdate).toHaveBeenLastCalledWith({});
    expect(screen.getByText("No aliases added yet. Add a new alias above.")).toBeInTheDocument();
  });

  it("does not submit the surrounding form when its own controls are used", async () => {
    const { onSubmit, user } = renderInForm();

    await user.click(screen.getByRole("button", { name: "Save team" }));
    expect(onSubmit).toHaveBeenCalledTimes(1);

    await addAlias(user, "fast-model", "gpt-4o");
    const row = screen.getByText("fast-model").closest("tr") as HTMLElement;
    await user.click(within(row).getByRole("button", { name: "Edit fast-model" }));
    await user.click(screen.getByRole("button", { name: "Cancel" }));
    await user.click(screen.getByRole("button", { name: "Delete fast-model" }));

    expect(onSubmit).toHaveBeenCalledTimes(1);
  });
});
