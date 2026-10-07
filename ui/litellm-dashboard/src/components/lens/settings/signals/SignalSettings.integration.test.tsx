import { fireEvent, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { renderWithLens } from "@/../tests/lens-test-utils";
import { testQueryClient } from "@/../tests/test-utils";

import type { SignalConfig } from "../../model/types";
import { SignalForm } from "./SignalSettings";

const saved: SignalConfig = {
  model: "jev",
  threshold: 0.5,
  signals: [{ id: "user_frustration", name: "User frustration", question: "Is the user frustrated?" }],
};

const updated: SignalConfig = {
  ...saved,
  model: "new-jev",
  threshold: 0.8,
};

const network = vi.fn<typeof fetch>();

describe("signal settings", () => {
  beforeEach(() => {
    testQueryClient.clear();
    network.mockReset();
    network.mockImplementation(async (input) => {
      const path = new URL(input instanceof Request ? input.url : String(input), "http://localhost").pathname;
      return Response.json(path === "/model_group/info" ? { data: [] } : {});
    });
    vi.stubGlobal("fetch", network);
  });

  it("updates a clean draft when saved settings change", async () => {
    const { rerender } = renderWithLens(<SignalForm saved={saved} />);
    const model = await screen.findByRole("combobox", { name: "System 1 model" });

    expect(model).toHaveValue("jev");
    rerender(<SignalForm saved={updated} />);

    expect(model).toHaveValue("new-jev");
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("keeps dirty edits until the user loads the latest settings", async () => {
    const user = userEvent.setup();
    const { rerender } = renderWithLens(<SignalForm saved={saved} />);
    const threshold = await screen.findByRole("spinbutton", { name: "Flag at score" });

    fireEvent.change(threshold, { target: { value: "70" } });
    rerender(<SignalForm saved={updated} />);

    expect(threshold).toHaveValue(70);
    expect(screen.getByRole("alert")).toHaveTextContent("Signals were changed elsewhere");
    await user.click(screen.getByRole("button", { name: "Load latest" }));

    expect(threshold).toHaveValue(80);
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("keeps a custom Tool failure signal separate while toggling the library signal", async () => {
    const user = userEvent.setup();
    const customQuestion = "Does this custom failure condition apply?";
    const customSignal = { id: "tool_failure", name: "Tool failure", question: customQuestion };
    const customConfig: SignalConfig = {
      ...saved,
      signals: [customSignal],
    };
    renderWithLens(<SignalForm saved={customConfig} />);

    const question = await screen.findByRole("textbox", { name: "Question for Tool failure" });
    const model = await screen.findByRole("combobox", { name: "System 1 model" });
    const threshold = screen.getByRole("spinbutton", { name: "Flag at score" });
    expect(question).toHaveValue(customQuestion);
    expect(screen.getByRole("button", { name: /^Tool failure/ })).toHaveAttribute("aria-pressed", "false");
    expect(model).toHaveValue("jev");
    expect(threshold).toHaveValue(50);

    await user.click(screen.getByRole("button", { name: /^Tool failure/ }));
    expect(screen.getByRole("button", { name: /^Tool failure/ })).toHaveAttribute("aria-pressed", "true");
    expect(question).toHaveValue(customQuestion);

    const editedQuestion = "Does this edited custom failure condition apply?";
    fireEvent.change(question, { target: { value: editedQuestion } });
    expect(question).toHaveValue(editedQuestion);
    expect(screen.getByRole("button", { name: /^Tool failure/ })).toHaveAttribute("aria-pressed", "true");

    await user.click(screen.getByRole("button", { name: "Remove Tool failure" }));
    expect(screen.queryByRole("textbox", { name: "Question for Tool failure" })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: /^Tool failure/ })).toHaveAttribute("aria-pressed", "true");

    await user.click(screen.getByRole("button", { name: /^Tool failure/ }));
    expect(screen.getByRole("button", { name: /^Tool failure/ })).toHaveAttribute("aria-pressed", "false");
    expect(screen.getByText("Pick at least one signal to flag traces")).toBeInTheDocument();
    expect(model).toHaveValue("jev");
    expect(threshold).toHaveValue(50);
  });

  it("removes only the library signal when toggling it off beside a custom signal", async () => {
    const user = userEvent.setup();
    const customQuestion = "Does this custom failure condition apply?";
    const customConfig: SignalConfig = {
      ...saved,
      signals: [{ id: "tool_failure", name: "Tool failure", question: customQuestion }],
    };
    renderWithLens(<SignalForm saved={customConfig} />);

    const question = await screen.findByRole("textbox", { name: "Question for Tool failure" });
    const tile = screen.getByRole("button", { name: /^Tool failure/ });

    await user.click(tile);
    expect(tile).toHaveAttribute("aria-pressed", "true");
    await user.click(tile);

    expect(screen.getByRole("button", { name: /^Tool failure/ })).toHaveAttribute("aria-pressed", "false");
    expect(question).toHaveValue(customQuestion);
  });
});
