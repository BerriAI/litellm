import { renderWithProviders, screen, waitFor } from "@/../tests/test-utils";
import userEvent from "@testing-library/user-event";
import type { OnUrlUpdateFunction } from "nuqs/adapters/testing";
import { describe, it, expect, vi } from "vitest";

import GuardrailTable from "./guardrail_table";
import { Guardrail, GuardrailDefinitionLocation } from "@/components/guardrails/types";

const baseProps = {
  isLoading: false,
  onDeleteClick: vi.fn(),
  onGuardrailClick: vi.fn(),
};

const makeGuardrail = (overrides: Partial<Guardrail> = {}): Guardrail => ({
  guardrail_id: "gr-1",
  guardrail_name: "PII Redaction",
  litellm_params: { guardrail: "presidio", mode: "pre_call", default_on: true },
  guardrail_info: null,
  created_at: "2021-01-01",
  updated_at: "2021-01-02",
  guardrail_definition_location: GuardrailDefinitionLocation.DB,
  ...overrides,
});

describe("GuardrailTable", () => {
  it("renders every column header", () => {
    renderWithProviders(<GuardrailTable guardrailsList={[]} {...baseProps} />);
    for (const header of ["Guardrail ID", "Name", "Provider", "Mode", "Default On", "Created At", "Updated At"]) {
      expect(screen.getByText(header)).toBeInTheDocument();
    }
  });

  it("renders the provider logo from the bundled guardrail logo map", () => {
    renderWithProviders(<GuardrailTable guardrailsList={[makeGuardrail()]} {...baseProps} />);
    const logo = screen.getByAltText("Presidio PII logo");
    expect(logo).toHaveAttribute("src", expect.stringContaining("microsoft_azure.svg"));
  });

  it("falls back to a letter avatar for an unknown provider slug", () => {
    const guardrail = makeGuardrail({
      litellm_params: { guardrail: "mystery_guard", mode: "pre_call", default_on: false },
    });
    renderWithProviders(<GuardrailTable guardrailsList={[guardrail]} {...baseProps} />);
    expect(screen.getByText("mystery_guard")).toBeInTheDocument();
    expect(screen.queryByAltText("mystery_guard logo")).not.toBeInTheDocument();
    expect(screen.getByText("m")).toBeInTheDocument();
  });

  it("renders a tag-based mode object instead of crashing the table", () => {
    const guardrail = makeGuardrail({
      litellm_params: {
        guardrail: "bedrock",
        mode: { tags: { "Service-Type: internal-service": "post_call" }, default: ["pre_call", "post_call"] },
        default_on: true,
      },
    });
    renderWithProviders(<GuardrailTable guardrailsList={[guardrail]} {...baseProps} />);
    expect(screen.getByText("pre_call, post_call (tag-based)")).toBeInTheDocument();
  });

  it("deletes a DB guardrail through the actions menu", async () => {
    const user = userEvent.setup();
    const onDeleteClick = vi.fn();
    const guardrail = makeGuardrail({ guardrail_id: "gr-9", guardrail_name: "Toxicity Filter" });
    renderWithProviders(<GuardrailTable guardrailsList={[guardrail]} {...baseProps} onDeleteClick={onDeleteClick} />);

    await user.click(screen.getByTestId("guardrail-actions-gr-9"));
    await user.click(await screen.findByTestId("guardrail-action-delete"));

    expect(onDeleteClick).toHaveBeenCalledWith("gr-9", "Toxicity Filter");
  });

  it("disables deletion for config guardrails so they cannot be removed from the dashboard", async () => {
    const user = userEvent.setup();
    const onDeleteClick = vi.fn();
    const guardrail = makeGuardrail({
      guardrail_id: "cfg-1",
      guardrail_name: "Config Guardrail",
      guardrail_definition_location: GuardrailDefinitionLocation.CONFIG,
    });
    renderWithProviders(<GuardrailTable guardrailsList={[guardrail]} {...baseProps} onDeleteClick={onDeleteClick} />);

    await user.click(screen.getByTestId("guardrail-actions-cfg-1"));
    const deleteItem = await screen.findByTestId("guardrail-action-delete");

    expect(deleteItem).toHaveAttribute("data-disabled");
    expect(onDeleteClick).not.toHaveBeenCalled();
  });

  describe("URL state", () => {
    const named = [
      makeGuardrail({ guardrail_id: "gr-b", guardrail_name: "name-bravo", created_at: "2024-03-01" }),
      makeGuardrail({ guardrail_id: "gr-c", guardrail_name: "name-charlie", created_at: "2024-01-01" }),
      makeGuardrail({ guardrail_id: "gr-a", guardrail_name: "name-alpha", created_at: "2024-02-01" }),
    ];
    const renderedNames = () => screen.getAllByTitle(/^name-/).map((cell) => cell.textContent);
    const manyGuardrails = Array.from({ length: 30 }, (_, index) =>
      makeGuardrail({ guardrail_id: `gr-${index}`, guardrail_name: `name-${index}` }),
    );
    const lastUrlUpdate = (onUrlUpdate: ReturnType<typeof vi.fn<OnUrlUpdateFunction>>) =>
      onUrlUpdate.mock.calls.at(-1)?.[0];

    it("sorts by newest created first when the URL has no sort", () => {
      renderWithProviders(<GuardrailTable guardrailsList={named} {...baseProps} />);
      expect(renderedNames()).toEqual(["name-bravo", "name-alpha", "name-charlie"]);
    });

    it("sorts by the column and direction in ?sort_by= and ?sort_order=", () => {
      renderWithProviders(<GuardrailTable guardrailsList={named} {...baseProps} />, {
        searchParams: "?sort_by=guardrail_name&sort_order=asc",
      });
      expect(renderedNames()).toEqual(["name-alpha", "name-bravo", "name-charlie"]);
    });

    it("falls back to the default sort for a column that cannot be sorted", () => {
      renderWithProviders(<GuardrailTable guardrailsList={named} {...baseProps} />, {
        searchParams: "?sort_by=provider&sort_order=desc",
      });
      expect(renderedNames()).toEqual(["name-bravo", "name-alpha", "name-charlie"]);
    });

    it("writes the clicked sort column and direction to the URL", async () => {
      const user = userEvent.setup();
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      renderWithProviders(<GuardrailTable guardrailsList={named} {...baseProps} />, { onUrlUpdate });

      await user.click(screen.getByTestId("sort-header-guardrail_name"));

      await waitFor(() => expect(lastUrlUpdate(onUrlUpdate)?.searchParams.get("sort_by")).toBe("guardrail_name"));
      expect(lastUrlUpdate(onUrlUpdate)?.searchParams.get("sort_order")).toBe("asc");
      expect(renderedNames()).toEqual(["name-alpha", "name-bravo", "name-charlie"]);
    });

    it("shows the page in ?page= and writes page changes back", async () => {
      const user = userEvent.setup();
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      renderWithProviders(<GuardrailTable guardrailsList={manyGuardrails} {...baseProps} />, {
        searchParams: "?page=2",
        onUrlUpdate,
      });

      expect(screen.getByTestId("pagination-page")).toHaveTextContent("Page 2 of 2");
      expect(screen.getAllByTitle(/^name-/)).toHaveLength(5);

      await user.click(screen.getByRole("button", { name: "Go to previous page" }));

      await waitFor(() => expect(lastUrlUpdate(onUrlUpdate)?.searchParams.has("page")).toBe(false));
      expect(screen.getByTestId("pagination-page")).toHaveTextContent("Page 1 of 2");
      expect(screen.getAllByTitle(/^name-/)).toHaveLength(25);

      await user.click(screen.getByRole("button", { name: "Go to next page" }));
      await waitFor(() => expect(lastUrlUpdate(onUrlUpdate)?.searchParams.get("page")).toBe("2"));
    });

    it("uses the page size in ?page_size=", () => {
      renderWithProviders(<GuardrailTable guardrailsList={manyGuardrails} {...baseProps} />, {
        searchParams: "?page_size=50",
      });
      expect(screen.getByTestId("pagination-page")).toHaveTextContent("Page 1 of 1");
      expect(screen.getAllByTitle(/^name-/)).toHaveLength(30);
    });
  });
});
