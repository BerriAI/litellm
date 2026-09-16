import userEvent from "@testing-library/user-event";
import type { OnUrlUpdateFunction } from "nuqs/adapters/testing";
import { describe, expect, it, Mock, vi } from "vitest";
import { Plugin } from "@/components/claude_code_plugins/types";
import { act, chooseSelectOption, fireEvent, renderWithProviders, screen, waitFor } from "../../../tests/test-utils";
import SkillHubDashboard from "./SkillHubDashboard";

const skill = (id: string, name: string, domain?: string): Plugin => ({
  id,
  name,
  description: `${name} description`,
  source: { source: "github", repo: `org/${name}` },
  domain,
  enabled: true,
});

const SKILLS = [skill("s-pdf", "pdf-tools", "Productivity"), skill("s-ledger", "ledger-sync", "Finance")];
const MANY_SKILLS = Array.from({ length: 30 }, (_, index) =>
  skill(`s-${index}`, `skill-${String(index).padStart(2, "0")}`),
);

const renderDashboard = (searchParams = "", skills: Plugin[] = SKILLS) => {
  const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
  const view = renderWithProviders(<SkillHubDashboard skills={skills} isLoading={false} publicPage />, {
    searchParams,
    onUrlUpdate,
  });
  return { ...view, onUrlUpdate, user: userEvent.setup() };
};

const lastUpdate = (onUrlUpdate: Mock<OnUrlUpdateFunction>) => {
  const event = onUrlUpdate.mock.calls.at(-1)?.[0];
  if (!event) throw new Error("no URL update was emitted");
  return event;
};

const settle = () => act(() => new Promise((resolve) => setTimeout(resolve, 20)));

const domainSelect = () => {
  const select = screen
    .getAllByRole("combobox")
    .find((element) => element.getAttribute("data-testid") !== "pagination-page-size");
  if (!select) throw new Error("domain select not rendered");
  return select;
};

const skillNames = () => screen.queryAllByRole("button", { name: /^(pdf-tools|ledger-sync|skill-\d+)$/ });

describe("SkillHubDashboard URL state", () => {
  it("filters the table by the skill_q search in the URL", () => {
    renderDashboard("?skill_q=ledger");

    expect(screen.getByPlaceholderText("Search by name, namespace, or tag…")).toHaveValue("ledger");
    expect(skillNames().map((button) => button.textContent)).toEqual(["ledger-sync"]);
  });

  it("writes the search box to skill_q", async () => {
    const { onUrlUpdate } = renderDashboard();

    fireEvent.change(screen.getByPlaceholderText("Search by name, namespace, or tag…"), { target: { value: "pdf" } });

    await waitFor(() => expect(lastUpdate(onUrlUpdate).searchParams.get("skill_q")).toBe("pdf"));
    expect(skillNames().map((button) => button.textContent)).toEqual(["pdf-tools"]);
  });

  it("filters the table by the skill_domain in the URL", () => {
    renderDashboard("?skill_domain=Productivity");

    expect(domainSelect()).toHaveTextContent("Productivity");
    expect(skillNames().map((button) => button.textContent)).toEqual(["pdf-tools"]);
  });

  it("writes the chosen domain to skill_domain and drops it for All Domains", async () => {
    const { onUrlUpdate, user } = renderDashboard();

    await chooseSelectOption(user, domainSelect(), "Finance");
    await waitFor(() => expect(lastUpdate(onUrlUpdate).searchParams.get("skill_domain")).toBe("Finance"));
    expect(skillNames().map((button) => button.textContent)).toEqual(["ledger-sync"]);

    await chooseSelectOption(user, domainSelect(), "All Domains");
    await waitFor(() => expect(lastUpdate(onUrlUpdate).searchParams.has("skill_domain")).toBe(false));
    expect(skillNames()).toHaveLength(2);
  });

  it("opens the skill named by ?skill= in place of the table", () => {
    renderDashboard("?skill=s-ledger");

    expect(screen.getByRole("heading", { name: "ledger-sync" })).toBeInTheDocument();
    expect(screen.queryByPlaceholderText("Search by name, namespace, or tag…")).not.toBeInTheDocument();
  });

  it("pushes ?skill= when a skill is opened and clears it when the detail's Back link is used", async () => {
    const { onUrlUpdate, user } = renderDashboard("?skill_q=pdf");

    await user.click(screen.getByRole("button", { name: "pdf-tools" }));

    await waitFor(() => expect(lastUpdate(onUrlUpdate).searchParams.get("skill")).toBe("s-pdf"));
    expect(lastUpdate(onUrlUpdate).options.history).toBe("push");
    expect(await screen.findByRole("heading", { name: "pdf-tools" })).toBeInTheDocument();

    await user.click(screen.getByText("Skills"));

    await waitFor(() => expect(lastUpdate(onUrlUpdate).searchParams.has("skill")).toBe(false));
    expect(lastUpdate(onUrlUpdate).searchParams.get("skill_q")).toBe("pdf");
    expect(await screen.findByPlaceholderText("Search by name, namespace, or tag…")).toBeInTheDocument();
  });

  it("sorts by the skill_sort_by and skill_sort_order in the URL", () => {
    renderDashboard("?skill_sort_by=name&skill_sort_order=desc");

    expect(skillNames().map((button) => button.textContent)).toEqual(["pdf-tools", "ledger-sync"]);
  });

  it("writes a header sort to skill_sort_by and skill_sort_order", async () => {
    const { onUrlUpdate, user } = renderDashboard();

    await user.click(screen.getByTestId("sort-header-domain"));

    await waitFor(() => expect(lastUpdate(onUrlUpdate).searchParams.get("skill_sort_by")).toBe("domain"));
    expect(skillNames().map((button) => button.textContent)).toEqual(["ledger-sync", "pdf-tools"]);

    await user.click(screen.getByTestId("sort-header-domain"));

    await waitFor(() => expect(lastUpdate(onUrlUpdate).searchParams.get("skill_sort_order")).toBe("desc"));
    expect(skillNames().map((button) => button.textContent)).toEqual(["pdf-tools", "ledger-sync"]);
  });

  it("writes the page to skill_page", async () => {
    const { onUrlUpdate, user } = renderDashboard("", MANY_SKILLS);

    await user.click(screen.getByTestId("pagination-next"));

    await waitFor(() => expect(lastUpdate(onUrlUpdate).searchParams.get("skill_page")).toBe("2"));
    expect(screen.getByTestId("pagination-page")).toHaveTextContent("Page 2 of 2");
  });

  it("keeps a deep-linked skill_page once the skills finish loading", async () => {
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    const { rerender } = renderWithProviders(<SkillHubDashboard skills={[]} isLoading={false} />, {
      searchParams: "?skill_page=2",
      onUrlUpdate,
    });
    await settle();
    rerender(<SkillHubDashboard skills={[]} isLoading />);
    await settle();
    rerender(<SkillHubDashboard skills={MANY_SKILLS} isLoading={false} />);
    await settle();

    expect(screen.getByRole("button", { name: "skill-29" })).toBeInTheDocument();
    expect(screen.getByTestId("pagination-page")).toHaveTextContent("Page 2 of 2");
    expect(onUrlUpdate).not.toHaveBeenCalled();
  });
});
