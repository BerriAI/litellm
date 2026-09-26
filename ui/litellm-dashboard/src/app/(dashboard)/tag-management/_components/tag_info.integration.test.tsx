import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { tagInfoCall, tagUpdateCall } from "@/components/networking";
import type { Tag } from "@/components/tag_management/types";

import TagInfoView from "./tag_info";

vi.mock("@/components/networking", () => ({
  tagInfoCall: vi.fn(),
  tagUpdateCall: vi.fn(),
  getProxyBaseUrl: () => "",
  getGlobalLitellmHeaderName: () => "Authorization",
  deriveErrorMessage: (errorData: unknown) => JSON.stringify(errorData),
  handleError: vi.fn(),
}));

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: () => ({ accessToken: "sk-test" }),
}));

vi.mock("@/components/organisms/create_key_button", () => ({
  fetchUserModels: vi.fn(
    (_userID: string, _userRole: string, _accessToken: string, setUserModels: (models: string[]) => void) => {
      setUserModels(["model-1", "model-2"]);
      return Promise.resolve();
    },
  ),
}));

const mockTagInfoCall = vi.mocked(tagInfoCall);
const mockTagUpdateCall = vi.mocked(tagUpdateCall);

const tag: Tag = {
  name: "prod-tag",
  description: "original description",
  models: ["model-1", "model-2"],
  model_info: { "model-1": "GPT-4", "model-2": "Claude-3" },
  created_at: "2024-01-01T00:00:00Z",
  updated_at: "2024-01-02T00:00:00Z",
  litellm_budget_table: { max_budget: 10, budget_duration: "7d", tpm_limit: 1000, rpm_limit: 60 },
};

const keyListFetch = vi.fn();

const renderTagInfo = (editTag: boolean) => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <QueryClientProvider client={queryClient}>
      <TagInfoView tagId="prod-tag" onClose={vi.fn()} accessToken="sk-test" is_admin editTag={editTag} />
    </QueryClientProvider>,
  );
};

const renderEditor = async () => {
  const user = userEvent.setup();
  renderTagInfo(true);
  const nameInput = await screen.findByLabelText("Tag Name");
  return { user, nameInput };
};

describe("TagInfoView save payload", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockTagInfoCall.mockResolvedValue({ "prod-tag": tag });
    mockTagUpdateCall.mockResolvedValue(undefined);
    keyListFetch.mockResolvedValue({
      ok: true,
      json: async () => ({ keys: [], total_count: 0, current_page: 1, total_pages: 0 }),
    });
    vi.stubGlobal("fetch", keyListFetch);
  });

  it("should send the edited fields and omit the budget fields while the budget section is collapsed", async () => {
    const { user, nameInput } = await renderEditor();

    await user.clear(nameInput);
    fireEvent.change(nameInput, { target: { value: "renamed-tag" } });

    const descriptionInput = screen.getByLabelText("Description");
    await user.clear(descriptionInput);
    fireEvent.change(descriptionInput, { target: { value: "updated description" } });

    await user.click(screen.getByRole("button", { name: "Save Changes" }));

    const expected = {
      name: "renamed-tag",
      description: "updated description",
      models: ["model-1", "model-2"],
      max_budget: undefined,
      tpm_limit: undefined,
      rpm_limit: undefined,
      budget_duration: undefined,
    };

    expect(mockTagUpdateCall).toHaveBeenCalledWith("sk-test", expected);
  });

  it("should send the budget fields once the budget section is expanded", async () => {
    const { user, nameInput } = await renderEditor();
    expect(nameInput).toHaveValue("prod-tag");

    await user.click(screen.getByRole("button", { name: /Budget & Rate Limits/ }));

    const maxBudgetInput = await screen.findByLabelText("Max Budget (USD)");
    await user.clear(maxBudgetInput);
    fireEvent.change(maxBudgetInput, { target: { value: "150.75" } });

    await user.click(screen.getByRole("button", { name: "Save Changes" }));

    const expected = {
      name: "prod-tag",
      description: "original description",
      models: ["model-1", "model-2"],
      max_budget: "150.75",
      tpm_limit: undefined,
      rpm_limit: undefined,
      budget_duration: "7d",
    };

    expect(mockTagUpdateCall).toHaveBeenCalledWith("sk-test", expected);
  });

  it("should block the save when the tag name is cleared", async () => {
    const { user, nameInput } = await renderEditor();

    await user.clear(nameInput);
    await user.click(screen.getByRole("button", { name: "Save Changes" }));

    expect(await screen.findByText("Please input a tag name")).toBeInTheDocument();
    expect(mockTagUpdateCall).not.toHaveBeenCalled();
  });

  it("keeps a typed budget when the section is collapsed and reopened, as antd's store did", async () => {
    const { user } = await renderEditor();
    const toggle = () => screen.getByRole("button", { name: /Budget & Rate Limits/ });

    await user.click(toggle());
    const maxBudgetInput = await screen.findByLabelText("Max Budget (USD)");
    await user.clear(maxBudgetInput);
    fireEvent.change(maxBudgetInput, { target: { value: "150.75" } });

    await user.click(toggle());
    await user.click(toggle());

    expect(await screen.findByLabelText("Max Budget (USD)")).toHaveValue(150.75);

    await user.click(screen.getByRole("button", { name: "Save Changes" }));

    const expected = {
      name: "prod-tag",
      description: "original description",
      models: ["model-1", "model-2"],
      max_budget: "150.75",
      tpm_limit: undefined,
      rpm_limit: undefined,
      budget_duration: "7d",
    };

    expect(mockTagUpdateCall).toHaveBeenCalledWith("sk-test", expected);
  });

  it("leaves the tag untouched and returns to the detail view when Cancel is clicked", async () => {
    const { user } = await renderEditor();

    const descriptionInput = screen.getByLabelText("Description");
    await user.clear(descriptionInput);
    fireEvent.change(descriptionInput, { target: { value: "abandoned description" } });

    await user.click(screen.getByRole("button", { name: "Cancel" }));

    expect(await screen.findByText("Tag Details")).toBeInTheDocument();
    expect(mockTagUpdateCall).not.toHaveBeenCalled();
  });
});

describe("TagInfoView virtual keys", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockTagInfoCall.mockResolvedValue({ "prod-tag": tag });
    vi.stubGlobal("fetch", keyListFetch);
  });

  it("should list the keys that /key/list returns for this tag, each linking to its key page", async () => {
    keyListFetch.mockResolvedValue({
      ok: true,
      json: async () => ({
        keys: [{ token: "tok-1", key_alias: "batch-key-1", key_name: "sk-...0001", team_id: "team-a", spend: 2 }],
        total_count: 1,
        current_page: 1,
        total_pages: 1,
      }),
    });
    renderTagInfo(false);

    expect(await screen.findByRole("link", { name: "batch-key-1" })).toHaveAttribute("href", "/ui/api-keys?key=tok-1");
    const requestUrl = new URL(keyListFetch.mock.calls[0][0], "http://localhost");
    expect(requestUrl.pathname).toBe("/key/list");
    expect(requestUrl.searchParams.get("tag")).toBe("prod-tag");
  });

  it("should say no virtual keys use the tag when /key/list returns none", async () => {
    keyListFetch.mockResolvedValue({
      ok: true,
      json: async () => ({ keys: [], total_count: 0, current_page: 1, total_pages: 0 }),
    });
    renderTagInfo(false);

    expect(await screen.findByText("No virtual keys use this tag")).toBeInTheDocument();
  });

  it("should show an error in the section when /key/list fails", async () => {
    keyListFetch.mockResolvedValue({ ok: false, json: async () => ({ error: "boom" }) });
    renderTagInfo(false);

    expect(await screen.findByText("Could not load the virtual keys for this tag")).toBeInTheDocument();
  });
});
