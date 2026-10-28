/* @vitest-environment jsdom */
import { act, fireEvent, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { setCallbacksCall } from "@/components/networking";
import { aliasDisplayStorage } from "./model_group_alias_ordering";
import ModelGroupAliasSettings from "./model_group_alias_settings";

vi.mock("@/components/networking", () => ({
  setCallbacksCall: vi.fn(),
}));

const BLOCKED_STORAGE = {
  getItem: () => null,
  setItem: () => {
    throw new Error("storage blocked");
  },
};

const aliases = {
  "fast-model": "claude-haiku",
  "smart-model": { model: "claude-sonnet", hidden: true },
};

const displayAliases = {
  zeta: "standard",
  "subscription/zeta": "standard",
  tester: "standard",
  alpha: "subscription/fast",
  reviewer: { model: "subscription/reasoning", hidden: true },
  "subscription/alpha": "standard",
};

const rowNames = () =>
  within(screen.getByRole("table"))
    .getAllByRole("row")
    .slice(1)
    .map((row) => within(row).getAllByRole("cell")[0].textContent);

describe("ModelGroupAliasSettings", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    localStorage.clear();
    vi.mocked(setCallbacksCall).mockResolvedValue({});
  });

  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  it("locks config-owned aliases while keeping their names and targets visible", async () => {
    render(
      <ModelGroupAliasSettings
        preferenceScope="test-user-proxy"
        accessToken="test-token"
        initialModelGroupAlias={aliases}
        managedByConfig
      />,
    );

    expect(
      screen.getByText(
        "These aliases are defined in config.yaml and are read-only here. Edit the config file to change them.",
      ),
    ).toBeInTheDocument();
    expect(screen.getByPlaceholderText("e.g., gpt-4o")).toBeDisabled();
    expect(screen.getByPlaceholderText("e.g., gpt-4o-mini-openai")).toBeDisabled();
    expect(screen.getByRole("button", { name: "Add Alias" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Edit alias fast-model" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Delete alias fast-model" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Edit alias smart-model" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Delete alias smart-model" })).toBeDisabled();

    expect(await screen.findByText("fast-model")).toBeInTheDocument();
    expect(screen.getByText("claude-haiku")).toBeInTheDocument();
    expect(screen.getByText("smart-model")).toBeInTheDocument();
    expect(screen.getByText("claude-sonnet")).toBeInTheDocument();

    await userEvent.setup().click(screen.getByRole("button", { name: "Delete alias fast-model" }));
    expect(setCallbacksCall).not.toHaveBeenCalled();
  });

  it("keeps aliases editable by default and saves the current string representation after delete", async () => {
    const user = userEvent.setup();
    render(
      <ModelGroupAliasSettings
        preferenceScope="test-user-proxy"
        accessToken="test-token"
        initialModelGroupAlias={aliases}
      />,
    );

    expect(await screen.findByText("fast-model")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Edit alias fast-model" })).toBeEnabled();
    expect(screen.getByRole("button", { name: "Delete alias fast-model" })).toBeEnabled();

    await user.click(screen.getByRole("button", { name: "Delete alias fast-model" }));

    expect(setCallbacksCall).toHaveBeenCalledWith("test-token", {
      router_settings: { model_group_alias: { "smart-model": "claude-sonnet" } },
    });
  });

  it("keeps ordered pins above sorted groups across remounts without changing routing", async () => {
    const user = userEvent.setup();
    const originalAliases = structuredClone(displayAliases);
    const props = {
      preferenceScope: "test-user-proxy",
      accessToken: "test-token",
      initialModelGroupAlias: displayAliases,
      managedByConfig: true,
    };
    const view = render(<ModelGroupAliasSettings {...props} />);

    await user.click(screen.getByRole("button", { name: "Pin alias tester" }));
    await user.click(screen.getByRole("button", { name: "Pin alias reviewer" }));
    await user.click(screen.getByRole("button", { name: "Move reviewer up" }));
    fireEvent.change(screen.getByLabelText("Place aliases with this prefix last"), {
      target: { value: "subscription/" },
    });
    expect(rowNames()).toEqual(["reviewer", "tester", "alpha", "zeta", "subscription/alpha", "subscription/zeta"]);

    await user.click(screen.getByRole("button", { name: "Alias Name" }));
    expect(rowNames()).toEqual(["reviewer", "tester", "zeta", "alpha", "subscription/zeta", "subscription/alpha"]);
    expect(screen.getByRole("columnheader", { name: "Alias Name" })).toHaveAttribute("aria-sort", "descending");
    await user.click(screen.getByRole("button", { name: "Target Model Group" }));
    expect(rowNames()).toEqual(["reviewer", "tester", "zeta", "alpha", "subscription/alpha", "subscription/zeta"]);
    await user.click(screen.getByRole("button", { name: "Target Model Group" }));
    expect(rowNames()).toEqual(["reviewer", "tester", "alpha", "zeta", "subscription/alpha", "subscription/zeta"]);
    view.unmount();
    render(<ModelGroupAliasSettings {...props} />);
    expect(rowNames()).toEqual(["reviewer", "tester", "alpha", "zeta", "subscription/alpha", "subscription/zeta"]);
    expect(screen.getByLabelText("Place aliases with this prefix last")).toHaveValue("subscription/");
    expect(screen.getByRole("button", { name: "Move reviewer up" })).toBeDisabled();
    expect(setCallbacksCall).not.toHaveBeenCalled();
    expect(displayAliases).toEqual(originalAliases);

    await user.click(screen.getByRole("button", { name: "Reset display order" }));
    expect(rowNames()).toEqual(["alpha", "reviewer", "subscription/alpha", "subscription/zeta", "tester", "zeta"]);
    expect(screen.getByRole("columnheader", { name: "Alias Name" })).toHaveAttribute("aria-sort", "ascending");
  });

  it.each([false, true])(
    "preserves pins through loading and successful edits (storage blocked: %s)",
    async (blocked) => {
      if (blocked) vi.stubGlobal("localStorage", BLOCKED_STORAGE);
      const user = userEvent.setup();
      const props = { preferenceScope: "test-user-proxy", accessToken: "test-token" };
      const view = render(<ModelGroupAliasSettings {...props} initialModelGroupAlias={displayAliases} />);
      await user.click(screen.getByRole("button", { name: "Pin alias tester" }));
      view.rerender(<ModelGroupAliasSettings {...props} initialModelGroupAlias={{}} />);
      view.rerender(<ModelGroupAliasSettings {...props} initialModelGroupAlias={displayAliases} />);
      expect(rowNames()[0]).toBe("tester");
      await user.click(screen.getByRole("button", { name: "Edit alias tester" }));
      const editRow = screen.getByRole("row", { name: /Save Cancel$/ });
      fireEvent.change(within(editRow).getAllByRole("textbox")[0], { target: { value: "renamed" } });
      vi.spyOn(console, "error").mockImplementation(() => {});
      vi.mocked(setCallbacksCall).mockRejectedValueOnce(new Error("save failed"));
      await user.click(screen.getByRole("button", { name: "Save" }));
      await user.click(screen.getByRole("button", { name: "Cancel" }));
      expect(screen.getByRole("button", { name: "Unpin alias tester" })).toBeInTheDocument();

      await user.click(screen.getByRole("button", { name: "Edit alias tester" }));
      const retryRow = screen.getByRole("row", { name: /Save Cancel$/ });
      fireEvent.change(within(retryRow).getAllByRole("textbox")[0], { target: { value: "renamed" } });
      const pendingSave = Promise.withResolvers<void>();
      vi.mocked(setCallbacksCall).mockReturnValueOnce(pendingSave.promise);
      await user.click(screen.getByRole("button", { name: "Save" }));
      fireEvent.change(screen.getByLabelText("Place aliases with this prefix last"), {
        target: { value: "subscription/" },
      });
      await act(async () => pendingSave.resolve());
      expect(screen.getByLabelText("Place aliases with this prefix last")).toHaveValue("subscription/");
      expect(rowNames()[0]).toBe("renamed");
      expect(screen.getByRole("button", { name: "Unpin alias renamed" })).toBeInTheDocument();
      await user.click(screen.getByRole("button", { name: "Delete alias renamed" }));
      fireEvent.change(screen.getByPlaceholderText("e.g., gpt-4o"), { target: { value: "renamed" } });
      fireEvent.change(screen.getByPlaceholderText("e.g., gpt-4o-mini-openai"), { target: { value: "standard" } });
      await user.click(screen.getByRole("button", { name: "Add Alias" }));
      expect(rowNames()[0]).toBe("alpha");
      expect(screen.getByRole("button", { name: "Pin alias renamed" })).toBeInTheDocument();
    },
  );

  it("falls back from corrupt preferences and isolates preferences by scope", async () => {
    const user = userEvent.setup();
    localStorage.setItem(aliasDisplayStorage("test-user-proxy").name, '{"sortBy":"unknown"}');
    const props = { accessToken: "test-token", initialModelGroupAlias: displayAliases };
    const view = render(<ModelGroupAliasSettings key="one" preferenceScope="test-user-proxy" {...props} />);
    expect(rowNames()[0]).toBe("alpha");
    await user.click(screen.getByRole("button", { name: "Pin alias tester" }));
    view.rerender(<ModelGroupAliasSettings key="two" preferenceScope="another-user-proxy" {...props} />);
    expect(rowNames()[0]).toBe("alpha");
    view.rerender(<ModelGroupAliasSettings key="one" preferenceScope="test-user-proxy" {...props} />);
    expect(rowNames()[0]).toBe("tester");
    view.rerender(<ModelGroupAliasSettings key="loading" preferenceScope={null} {...props} />);
    expect(screen.getByRole("button", { name: "Pin alias tester" })).toBeDisabled();
  });

  it("keeps display changes usable when browser storage refuses writes", async () => {
    const user = userEvent.setup();
    vi.stubGlobal("localStorage", BLOCKED_STORAGE);
    render(
      <ModelGroupAliasSettings
        preferenceScope="blocked-storage"
        accessToken="test-token"
        initialModelGroupAlias={displayAliases}
      />,
    );
    await user.click(screen.getByRole("button", { name: "Pin alias tester" }));
    expect(rowNames()[0]).toBe("tester");
    expect(screen.getByRole("status")).toHaveTextContent("could not be saved");
    await user.click(screen.getByRole("button", { name: "Unpin alias tester" }));
    expect(rowNames()[0]).toBe("alpha");
    expect(setCallbacksCall).not.toHaveBeenCalled();
  });
});
