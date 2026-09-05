import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import React from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const { GET, PUT, DELETE, POST, modelInfoCall, userRole } = vi.hoisted(() => ({
  GET: vi.fn(),
  PUT: vi.fn(),
  DELETE: vi.fn(),
  POST: vi.fn(),
  modelInfoCall: vi.fn(),
  userRole: { current: "Admin" },
}));
vi.mock("@/lib/http/api", () => ({ fetchClient: { GET, PUT, DELETE, POST } }));
vi.mock("@/components/networking", () => ({
  modelInfoCall,
  modelHubCall: vi.fn(),
  modelAvailableCall: vi.fn(),
}));

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: () => ({ accessToken: "sk-test", userId: "user-1", userRole: userRole.current }),
}));

import AccessGroupBudgetsPanel from "./AccessGroupBudgetsPanel";

const BUDGETED_GROUP = {
  access_group: "premium",
  model_names: ["premium-nano"],
  deployment_count: 1,
  spend: 1.25,
  budget: {
    budget_id: "budget-1",
    max_budget: 2.5,
    soft_budget: null,
    budget_duration: "30d",
    budget_reset_at: null,
  },
};

const FREE_GROUP = {
  access_group: "shared",
  model_names: ["shared-nano"],
  deployment_count: 2,
  spend: 0,
  budget: null,
};

const DATABASE_DEPLOYMENT = { model_name: "gold-nano", model_info: { id: "m-1", db_model: true } };
const CONFIG_DEPLOYMENT = { model_name: "config-nano", model_info: { id: "m-2", db_model: false } };
const MODEL_INFO_PAGE = {
  data: [DATABASE_DEPLOYMENT, CONFIG_DEPLOYMENT],
  total_count: 2,
  current_page: 1,
  total_pages: 1,
  size: 1000,
};

const renderPanel = () => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={queryClient}>
      <AccessGroupBudgetsPanel />
    </QueryClientProvider>,
  );
};

const openActions = async (accessGroup: string) => {
  await userEvent.click(await screen.findByTestId(`access-group-actions-${accessGroup}`));
};

describe("AccessGroupBudgetsPanel", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    userRole.current = "Admin";
    GET.mockResolvedValue({ data: { access_groups: [BUDGETED_GROUP, FREE_GROUP] } });
    PUT.mockResolvedValue({ data: { access_group: "shared", spend: 0, budget: null } });
    DELETE.mockResolvedValue({ data: { access_group: "premium", budget_deleted: true, message: "ok" } });
    POST.mockResolvedValue({ data: { access_group: "gold", model_names: ["gold-nano"], models_updated: 1 } });
    modelInfoCall.mockResolvedValue(MODEL_INFO_PAGE);
  });

  it("lists each group with the spend drawn against its shared budget", async () => {
    renderPanel();

    expect(await screen.findByText("premium")).toBeInTheDocument();
    expect(screen.getByText("$1.2500")).toBeInTheDocument();
    expect(screen.getByText("of $2.50")).toBeInTheDocument();
    expect(screen.getByText("monthly")).toBeInTheDocument();
    expect(GET).toHaveBeenCalledWith("/access_group/list");
  });

  it("keeps a sub-cent budget readable instead of rounding it away to $0.00", async () => {
    GET.mockResolvedValue({
      data: { access_groups: [{ ...BUDGETED_GROUP, budget: { ...BUDGETED_GROUP.budget, max_budget: 0.00002 } }] },
    });
    renderPanel();

    expect(await screen.findByText("of $0.00002")).toBeInTheDocument();
  });

  it("shows a group with no budget as unlimited and offers nothing to clear", async () => {
    renderPanel();

    expect(await screen.findByText("· Unlimited")).toBeInTheDocument();

    await openActions("shared");

    expect(await screen.findByText("Set budget")).toBeInTheDocument();
    expect(screen.getByTestId("access-group-action-clear-budget")).toHaveAttribute("aria-disabled", "true");
  });

  it("sends the filled fields to the group's budget route", async () => {
    renderPanel();
    await openActions("shared");
    await userEvent.click(await screen.findByText("Set budget"));

    fireEvent.change(await screen.findByLabelText(/Max Budget/), { target: { value: "12.5" } });
    await userEvent.click(screen.getByRole("button", { name: "Save Budget" }));

    await waitFor(() =>
      expect(PUT).toHaveBeenCalledWith("/access_group/{access_group}/budget", {
        params: { path: { access_group: "shared" } },
        body: { max_budget: 12.5 },
      }),
    );
  });

  it("pre-fills the modal from the budget the group already has", async () => {
    renderPanel();
    await openActions("premium");
    await userEvent.click(await screen.findByText("Edit budget"));

    expect(await screen.findByLabelText(/Max Budget/)).toHaveValue(2.5);
  });

  it("refuses to save a budget with every field blank", async () => {
    renderPanel();
    await openActions("shared");
    await userEvent.click(await screen.findByText("Set budget"));
    await userEvent.click(await screen.findByRole("button", { name: "Save Budget" }));

    expect(await screen.findByText(/Set at least one of max budget/)).toBeInTheDocument();
    expect(PUT).not.toHaveBeenCalled();
  });

  it("offers an admin viewer no way to start a write the proxy would reject with a 403", async () => {
    userRole.current = "Admin Viewer";
    renderPanel();

    expect(await screen.findByText("premium")).toBeInTheDocument();

    await openActions("premium");

    expect(await screen.findByTestId("access-group-action-set-budget")).toHaveAttribute("aria-disabled", "true");
    expect(screen.getByTestId("access-group-action-clear-budget")).toHaveAttribute("aria-disabled", "true");
  });

  it("does not offer a budget on a group whose name a path segment cannot carry", async () => {
    GET.mockResolvedValue({ data: { access_groups: [{ ...FREE_GROUP, access_group: "openai/prod" }] } });
    renderPanel();

    await openActions("openai/prod");

    expect(await screen.findByTestId("access-group-action-set-budget")).toHaveAttribute("aria-disabled", "true");
  });

  it("clears a budget only after the confirmation is accepted", async () => {
    renderPanel();
    await openActions("premium");
    await userEvent.click(await screen.findByRole("menuitem", { name: /clear budget/i }));

    expect(DELETE).not.toHaveBeenCalled();

    await userEvent.click(await screen.findByRole("button", { name: /^delete$/i }));

    await waitFor(() =>
      expect(DELETE).toHaveBeenCalledWith("/access_group/{access_group}/budget", {
        params: { path: { access_group: "premium" } },
      }),
    );
  });

  describe("creating a group", () => {
    const openCreate = async () => {
      await userEvent.click(await screen.findByTestId("create-access-group"));
      await screen.findByRole("dialog");
    };

    const nameInput = () => screen.getByLabelText(/access group name/i);
    const submitButton = () => within(screen.getByRole("dialog")).getByRole("button", { name: /create access group/i });

    // Base UI marks the dialog inert while the option list is open, so dismiss it before touching the form
    const pickModel = async (modelName: string) => {
      await userEvent.click(within(screen.getByRole("dialog")).getByRole("combobox"));
      await userEvent.click(await screen.findByText(modelName));
      await userEvent.click(nameInput());
    };

    it("hides creation from an admin viewer", async () => {
      userRole.current = "Admin Viewer";
      renderPanel();

      expect(await screen.findByText("premium")).toBeInTheDocument();
      expect(screen.queryByTestId("create-access-group")).not.toBeInTheDocument();
    });

    it("offers only models that have a database deployment", async () => {
      renderPanel();
      await openCreate();

      await userEvent.click(within(screen.getByRole("dialog")).getByRole("combobox"));

      expect(await screen.findByText("gold-nano")).toBeInTheDocument();
      expect(screen.queryByText("config-nano")).not.toBeInTheDocument();
    });

    it("tags the chosen models with the trimmed name and lists the new group", async () => {
      renderPanel();
      await openCreate();

      fireEvent.change(nameInput(), { target: { value: " gold " } });
      await pickModel("gold-nano");
      GET.mockResolvedValue({
        data: { access_groups: [BUDGETED_GROUP, FREE_GROUP, { ...FREE_GROUP, access_group: "gold" }] },
      });
      await userEvent.click(submitButton());

      await waitFor(() =>
        expect(POST).toHaveBeenCalledWith("/access_group/new", {
          body: { access_group: "gold", model_names: ["gold-nano"] },
        }),
      );
      expect(await screen.findByText("gold")).toBeInTheDocument();
      expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    });

    it("refuses a name the proxy already has without sending a request", async () => {
      renderPanel();
      await openCreate();

      fireEvent.change(nameInput(), { target: { value: "premium" } });
      await pickModel("gold-nano");
      await userEvent.click(submitButton());

      expect(await screen.findByText("An access group with this name already exists")).toBeInTheDocument();
      expect(POST).not.toHaveBeenCalled();
    });

    it("requires at least one model", async () => {
      renderPanel();
      await openCreate();

      fireEvent.change(nameInput(), { target: { value: "gold" } });
      await userEvent.click(submitButton());

      expect(await screen.findByText("Pick at least one model")).toBeInTheDocument();
      expect(POST).not.toHaveBeenCalled();
    });
  });
});
