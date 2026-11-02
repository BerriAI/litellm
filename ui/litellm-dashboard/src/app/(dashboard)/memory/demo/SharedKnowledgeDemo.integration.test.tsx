import { act, fireEvent, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { OnUrlUpdateFunction } from "nuqs/adapters/testing";
import { describe, expect, it, vi } from "vitest";
import { uiHref } from "@/utils/uiHref";
import { renderWithProviders } from "../../../../../tests/test-utils";
import SharedKnowledgeDemo from "./SharedKnowledgeDemo";

const initialTitle = "Exclude inactive accounts from cross-account totals";
const sampleNotice = "Using a published sample memo. Save your draft in step 2 to use it here.";

async function chooseStep(user: ReturnType<typeof userEvent.setup>, name: RegExp) {
  const step = within(screen.getByRole("navigation", { name: "Demo steps" })).getByRole("button", { name });
  await user.click(step);
  await waitFor(() => expect(step).toHaveAttribute("aria-current", "step"));
}

describe("SharedKnowledgeDemo", () => {
  it("should render the knowledge categories with a link back to Memory", () => {
    renderWithProviders(<SharedKnowledgeDemo />);

    expect(screen.getByRole("heading", { name: "Shared knowledge", exact: true })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Back to Memory" })).toHaveAttribute("href", uiHref("memory"));
    expect(screen.getByRole("switch", { name: "Enable Account reporting" })).toBeChecked();
    expect(screen.getByText("User systems", { exact: true })).toBeInTheDocument();
    expect(screen.getByText("Operator systems", { exact: true })).toBeInTheDocument();
  });

  it("should carry a scoped memo through publication, reuse, feedback, and skill promotion", async () => {
    const user = userEvent.setup();
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    const title = "Use active child accounts in reporting totals";
    renderWithProviders(<SharedKnowledgeDemo />, { searchParams: "?workspace=demo", onUrlUpdate });

    await user.click(screen.getByRole("switch", { name: "Enable Account reporting" }));
    await chooseStep(user, /Capture a memo/);
    expect(screen.getByRole("button", { name: "Save Memo to Shared Knowledge" })).toBeDisabled();
    expect(screen.getByRole("alert")).toHaveTextContent("Enable Account reporting in step 1 to save this memo.");
    await chooseStep(user, /Define scope/);
    await user.click(screen.getByRole("switch", { name: "Enable Account reporting" }));
    await chooseStep(user, /Capture a memo/);
    act(() => {
      fireEvent.change(screen.getByRole("textbox", { name: "Title", exact: true }), { target: { value: title } });
      fireEvent.change(screen.getByRole("textbox", { name: "Summary", exact: true }), {
        target: { value: "Filter active parent and child accounts before calculating the total." },
      });
    });
    await user.click(screen.getByRole("button", { name: "Save Memo to Shared Knowledge" }));
    expect(screen.getByRole("heading", { name: "Published memo", exact: true })).toBeInTheDocument();

    await chooseStep(user, /Review & reuse/);
    expect(screen.getByRole("heading", { name: title, exact: true })).toBeInTheDocument();
    expect(screen.queryByText(sampleNotice)).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Run Judge Review" }));
    expect(screen.getByText("Accepted · v2 published", { exact: true })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Evidence-backed Update" })).toBeDisabled();
    await user.click(screen.getByRole("button", { name: "Load v2 into Session" }));
    await user.click(screen.getByRole("button", { name: "Record Helpful Outcome" }));
    expect(screen.getByRole("button", { name: "Outcome Recorded" })).toBeDisabled();

    await chooseStep(user, /Learn & curate/);
    const memoRow = screen.getByRole("row", { name: new RegExp(title) });
    expect(memoRow).toHaveTextContent("29 helpful / 31 known");
    expect(memoRow).toHaveTextContent("33 loads · 8 agents");
    await user.click(screen.getByRole("button", { name: "Run Periodic Review" }));
    expect(memoRow).toHaveTextContent("Promoted to skill");
    expect(screen.getByRole("row", { name: /Legacy account-total workaround/ })).toHaveTextContent("Archived");
    expect(screen.getByRole("row", { name: /Bazel: recover from a corrupted JAR/ })).toHaveTextContent("Retained");
    await user.click(screen.getByRole("button", { name: "View Skill" }));
    expect(screen.getByRole("dialog", { name: "Account reporting checks" })).toHaveTextContent(
      "Derived from memo_014 · v2",
    );
    await user.click(screen.getByRole("button", { name: "Close Skill" }));

    await waitFor(() => expect(onUrlUpdate.mock.calls.at(-1)?.[0].searchParams.get("step")).toBe("4"));
    expect(onUrlUpdate.mock.calls.at(-1)?.[0].searchParams.get("workspace")).toBe("demo");
    expect(onUrlUpdate.mock.calls.at(-1)?.[0].options.history).toBe("push");
  });

  it("should preserve a published version and recorded feedback when a later correction is rejected", async () => {
    const user = userEvent.setup();
    renderWithProviders(<SharedKnowledgeDemo />, { searchParams: "?step=3" });
    await user.click(screen.getByRole("button", { name: "Run Judge Review" }));
    await user.click(screen.getByRole("button", { name: "Load v2 into Session" }));
    await user.click(screen.getByRole("button", { name: "Record Helpful Outcome" }));
    act(() => {
      fireEvent.change(screen.getByRole("textbox", { name: "Search shared knowledge" }), {
        target: { value: "unrelated quantum penguins" },
      });
    });
    await user.click(screen.getByRole("button", { name: "Search", exact: true }));
    expect(screen.getByText("No matching demo memos", { exact: true })).toBeInTheDocument();
    act(() => {
      fireEvent.change(screen.getByRole("textbox", { name: "Search shared knowledge" }), {
        target: { value: "inactive accounts" },
      });
    });
    await user.click(screen.getByRole("button", { name: "Search", exact: true }));
    await user.click(screen.getByRole("button", { name: "Try Unsupported Update" }));
    await user.click(screen.getByRole("button", { name: "Run Judge Review" }));

    expect(screen.getByText("Rejected · v2 remains published", { exact: true })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Load v2 into Session" })).toBeEnabled();
    expect(screen.getByRole("button", { name: "Evidence-backed Update" })).toBeDisabled();
    await chooseStep(user, /Learn & curate/);
    expect(screen.getByRole("row", { name: new RegExp(initialTitle) })).toHaveTextContent("29 helpful / 31 known");
  });

  it("should keep an unsaved draft separate and preserve the version used to promote a skill", async () => {
    const user = userEvent.setup();
    const draftTitle = "Unpublished reporting investigation";
    renderWithProviders(<SharedKnowledgeDemo />, { searchParams: "?step=2" });
    act(() => {
      fireEvent.change(screen.getByRole("textbox", { name: "Title", exact: true }), { target: { value: draftTitle } });
    });
    await chooseStep(user, /Review & reuse/);
    expect(screen.getByText(sampleNotice)).toBeInTheDocument();
    expect(screen.queryByText(draftTitle)).not.toBeInTheDocument();
    expect(screen.getByRole("heading", { name: initialTitle, exact: true })).toBeInTheDocument();
    await chooseStep(user, /Learn & curate/);
    await user.click(screen.getByRole("button", { name: "Run Periodic Review" }));
    await chooseStep(user, /Review & reuse/);
    await user.click(screen.getByRole("button", { name: "Run Judge Review" }));
    await chooseStep(user, /Learn & curate/);
    await user.click(screen.getByRole("button", { name: "View Skill" }));

    expect(screen.getByRole("dialog", { name: "Account reporting checks" })).toHaveTextContent(
      "Derived from memo_014 · v1",
    );
    expect(screen.getByRole("dialog", { name: "Account reporting checks" })).toHaveTextContent(
      "Explicitly exclude inactive accounts from the total.",
    );
    await user.click(screen.getByRole("button", { name: "Close Skill" }));
    await chooseStep(user, /Capture a memo/);
    await user.click(screen.getByRole("button", { name: "Save Memo to Shared Knowledge" }));
    await chooseStep(user, /Review & reuse/);
    expect(screen.getByRole("heading", { name: draftTitle, exact: true })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Load v1 into Session" })).toBeEnabled();
  });

  it.each([
    ["2", "Turn a hard-won finding into a memo"],
    ["3", "Review corrections. Share the better answer."],
    ["4", "Let outcomes improve the knowledge base"],
    ["99", "Define what agents should remember"],
  ])("should open the valid step or fall back to scope for step=%s", (step, heading) => {
    renderWithProviders(<SharedKnowledgeDemo />, { searchParams: `?step=${step}` });

    expect(screen.getByRole("heading", { name: heading, exact: true })).toBeInTheDocument();
  });
});
