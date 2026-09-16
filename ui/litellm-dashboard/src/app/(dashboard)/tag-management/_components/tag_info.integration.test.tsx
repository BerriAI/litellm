import { fireEvent, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { OnUrlUpdateFunction } from "nuqs/adapters/testing";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { tagInfoCall, tagUpdateCall } from "@/components/networking";
import type { Tag } from "@/components/tag_management/types";

import { renderWithProviders } from "@/../tests/test-utils";
import TagInfoView from "./tag_info";

vi.mock("@/components/networking", () => ({
  tagInfoCall: vi.fn(),
  tagUpdateCall: vi.fn(),
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

type UrlUpdateMock = ReturnType<typeof vi.fn<OnUrlUpdateFunction>>;

const lastUrlUpdate = (onUrlUpdate: UrlUpdateMock) => onUrlUpdate.mock.calls.at(-1)?.[0];

const renderTagInfo = (searchParams: string, onUrlUpdate?: UrlUpdateMock) =>
  renderWithProviders(<TagInfoView tagId="prod-tag" onClose={vi.fn()} accessToken="sk-test" is_admin />, {
    searchParams,
    onUrlUpdate,
  });

const renderEditor = async (onUrlUpdate?: UrlUpdateMock) => {
  const user = userEvent.setup();
  renderTagInfo("?tag=prod-tag&edit=true", onUrlUpdate);
  const nameInput = await screen.findByLabelText("Tag Name");
  return { user, nameInput };
};

describe("TagInfoView save payload", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockTagInfoCall.mockResolvedValue({ "prod-tag": tag });
    mockTagUpdateCall.mockResolvedValue(undefined);
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
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    const { user } = await renderEditor(onUrlUpdate);

    const descriptionInput = screen.getByLabelText("Description");
    await user.clear(descriptionInput);
    fireEvent.change(descriptionInput, { target: { value: "abandoned description" } });

    await user.click(screen.getByRole("button", { name: "Cancel" }));

    expect(await screen.findByText("Tag Details")).toBeInTheDocument();
    expect(mockTagUpdateCall).not.toHaveBeenCalled();
    await waitFor(() => expect(onUrlUpdate).toHaveBeenCalled());
    expect(lastUrlUpdate(onUrlUpdate)?.searchParams.has("edit")).toBe(false);
    expect(lastUrlUpdate(onUrlUpdate)?.searchParams.get("tag")).toBe("prod-tag");
  });
});

describe("TagInfoView ?edit= mode", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockTagInfoCall.mockResolvedValue({ "prod-tag": tag });
    mockTagUpdateCall.mockResolvedValue(undefined);
  });

  it("shows the read-only details when ?edit= is absent", async () => {
    renderTagInfo("?tag=prod-tag");

    expect(await screen.findByText("Tag Details")).toBeInTheDocument();
    expect(screen.queryByLabelText("Tag Name")).not.toBeInTheDocument();
  });

  it("writes ?edit=true without a new history entry when Edit Tag is clicked", async () => {
    const user = userEvent.setup();
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    renderTagInfo("?tag=prod-tag", onUrlUpdate);

    await user.click(await screen.findByRole("button", { name: "Edit Tag" }));

    expect(await screen.findByLabelText("Tag Name")).toHaveValue("prod-tag");
    await waitFor(() => expect(lastUrlUpdate(onUrlUpdate)?.searchParams.get("edit")).toBe("true"));
    expect(lastUrlUpdate(onUrlUpdate)?.options.history).toBe("replace");
    expect(lastUrlUpdate(onUrlUpdate)?.searchParams.get("tag")).toBe("prod-tag");
  });

  it("seeds the saved budget when the link opens straight into edit mode", async () => {
    const { user } = await renderEditor();

    await user.click(screen.getByRole("button", { name: /Budget & Rate Limits/ }));

    expect(await screen.findByLabelText("Max Budget (USD)")).toHaveValue(10);
  });

  it("leaves the budget unseeded when edit mode is entered from the detail view", async () => {
    const user = userEvent.setup();
    renderTagInfo("?tag=prod-tag");

    await user.click(await screen.findByRole("button", { name: "Edit Tag" }));
    await user.click(await screen.findByRole("button", { name: /Budget & Rate Limits/ }));

    expect(await screen.findByLabelText("Max Budget (USD)")).toHaveValue(null);
  });

  it("clears ?edit= after a successful save", async () => {
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    const { user } = await renderEditor(onUrlUpdate);

    await user.click(screen.getByRole("button", { name: "Save Changes" }));

    await waitFor(() => expect(mockTagUpdateCall).toHaveBeenCalled());
    await waitFor(() => expect(lastUrlUpdate(onUrlUpdate)?.searchParams.has("edit")).toBe(false));
    expect(await screen.findByText("Tag Details")).toBeInTheDocument();
  });
});
