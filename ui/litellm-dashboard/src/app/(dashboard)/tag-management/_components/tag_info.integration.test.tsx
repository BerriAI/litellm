import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { tagInfoCall, tagUpdateCall } from "@/components/networking";
import type { Tag } from "@/components/tag_management/types";

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

const renderEditor = async () => {
  const user = userEvent.setup();
  render(<TagInfoView tagId="prod-tag" onClose={vi.fn()} accessToken="sk-test" is_admin editTag />);
  const nameInput = await screen.findByLabelText("Tag Name");
  return { user, nameInput };
};

describe("TagInfoView save payload", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockTagInfoCall.mockResolvedValue({ "prod-tag": tag });
    mockTagUpdateCall.mockResolvedValue(undefined);
  });

  it("leaves the budget unchanged while the budget section is collapsed", async () => {
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
      max_budget: 150.75,
      tpm_limit: 1000,
      rpm_limit: 60,
      budget_duration: "7d",
    };

    expect(mockTagUpdateCall).toHaveBeenCalledWith("sk-test", expected);
  });

  it("sends null when the budget is cleared", async () => {
    const { user } = await renderEditor();
    await user.click(screen.getByRole("button", { name: /Budget & Rate Limits/ }));
    const maxBudgetInput = await screen.findByLabelText("Max Budget (USD)");
    fireEvent.change(maxBudgetInput, { target: { value: "" } });
    await user.click(screen.getByRole("button", { name: "Save Changes" }));
    expect(mockTagUpdateCall).toHaveBeenCalledWith("sk-test", expect.objectContaining({ max_budget: null }));
  });

  it("sends zero as a budget value", async () => {
    const { user } = await renderEditor();
    await user.click(screen.getByRole("button", { name: /Budget & Rate Limits/ }));
    fireEvent.change(await screen.findByLabelText("Max Budget (USD)"), { target: { value: "0" } });
    await user.click(screen.getByRole("button", { name: "Save Changes" }));
    expect(mockTagUpdateCall).toHaveBeenCalledWith("sk-test", expect.objectContaining({ max_budget: 0 }));
  });

  it("sends the prefilled budget when it is left unchanged", async () => {
    const { user } = await renderEditor();
    await user.click(screen.getByRole("button", { name: /Budget & Rate Limits/ }));
    expect(await screen.findByLabelText("Max Budget (USD)")).toHaveValue(10);
    await user.click(screen.getByRole("button", { name: "Save Changes" }));
    expect(mockTagUpdateCall).toHaveBeenCalledWith("sk-test", expect.objectContaining({ max_budget: 10 }));
  });

  it("blocks negative budgets before sending the request", async () => {
    const { user } = await renderEditor();
    await user.click(screen.getByRole("button", { name: /Budget & Rate Limits/ }));
    fireEvent.change(await screen.findByLabelText("Max Budget (USD)"), { target: { value: "-1" } });
    await user.click(screen.getByRole("button", { name: "Save Changes" }));
    expect(await screen.findByText("Enter a nonnegative budget")).toBeInTheDocument();
    expect(mockTagUpdateCall).not.toHaveBeenCalled();
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
      max_budget: 150.75,
      tpm_limit: 1000,
      rpm_limit: 60,
      budget_duration: "7d",
    };

    expect(mockTagUpdateCall).toHaveBeenCalledWith("sk-test", expected);
  });

  it("sends the edited TPM and RPM limits as integers", async () => {
    const { user } = await renderEditor();
    await user.click(screen.getByRole("button", { name: /Budget & Rate Limits/ }));

    expect(await screen.findByLabelText("TPM Limit")).toHaveValue(1000);
    expect(screen.getByLabelText("RPM Limit")).toHaveValue(60);

    fireEvent.change(screen.getByLabelText("TPM Limit"), { target: { value: "50000" } });
    fireEvent.change(screen.getByLabelText("RPM Limit"), { target: { value: "120" } });
    await user.click(screen.getByRole("button", { name: "Save Changes" }));

    const expected = { tpm_limit: 50000, rpm_limit: 120, max_budget: 10, budget_duration: "7d" };
    expect(mockTagUpdateCall).toHaveBeenCalledWith("sk-test", expect.objectContaining(expected));
  });

  it("sends null for a cleared TPM limit so the proxy drops it", async () => {
    const { user } = await renderEditor();
    await user.click(screen.getByRole("button", { name: /Budget & Rate Limits/ }));

    fireEvent.change(await screen.findByLabelText("TPM Limit"), { target: { value: "" } });
    await user.click(screen.getByRole("button", { name: "Save Changes" }));

    expect(mockTagUpdateCall).toHaveBeenCalledWith(
      "sk-test",
      expect.objectContaining({ tpm_limit: null, rpm_limit: 60 }),
    );
  });

  it("rejects a fractional or negative RPM limit", async () => {
    const { user } = await renderEditor();
    await user.click(screen.getByRole("button", { name: /Budget & Rate Limits/ }));

    fireEvent.change(await screen.findByLabelText("RPM Limit"), { target: { value: "1.5" } });
    await user.click(screen.getByRole("button", { name: "Save Changes" }));
    expect(await screen.findByText("Enter a whole number of 0 or more")).toBeInTheDocument();

    fireEvent.change(screen.getByLabelText("RPM Limit"), { target: { value: "-1" } });
    await user.click(screen.getByRole("button", { name: "Save Changes" }));
    expect(await screen.findByText("Enter a whole number of 0 or more")).toBeInTheDocument();
    expect(mockTagUpdateCall).not.toHaveBeenCalled();
  });

  it("shows the current limits when editing from the Edit Tag button rather than the row action", async () => {
    const user = userEvent.setup();
    render(<TagInfoView tagId="prod-tag" onClose={vi.fn()} accessToken="sk-test" is_admin editTag={false} />);

    await user.click(await screen.findByRole("button", { name: "Edit Tag" }));
    await user.click(screen.getByRole("button", { name: /Budget & Rate Limits/ }));

    expect(await screen.findByLabelText("TPM Limit")).toHaveValue(1000);
    expect(screen.getByLabelText("RPM Limit")).toHaveValue(60);
    expect(screen.getByLabelText("Max Budget (USD)")).toHaveValue(10);
  });

  it("saves a tag whose budget row has null fields instead of failing validation on the nulls", async () => {
    mockTagInfoCall.mockResolvedValue({
      "prod-tag": {
        ...tag,
        litellm_budget_table: { max_budget: null, budget_duration: null, tpm_limit: 1000, rpm_limit: null },
      },
    });
    const { user } = await renderEditor();
    await user.click(screen.getByRole("button", { name: /Budget & Rate Limits/ }));

    fireEvent.change(await screen.findByLabelText("RPM Limit"), { target: { value: "30" } });
    await user.click(screen.getByRole("button", { name: "Save Changes" }));

    expect(screen.queryByText(/Invalid input/)).not.toBeInTheDocument();
    const expected = { max_budget: null, tpm_limit: 1000, rpm_limit: 30, budget_duration: null };
    expect(mockTagUpdateCall).toHaveBeenCalledWith("sk-test", expect.objectContaining(expected));
  });

  it("saves a tag that has no description instead of failing validation on the null", async () => {
    mockTagInfoCall.mockResolvedValue({ "prod-tag": { ...tag, description: null } });
    const { user } = await renderEditor();
    await user.click(screen.getByRole("button", { name: /Budget & Rate Limits/ }));

    fireEvent.change(await screen.findByLabelText("RPM Limit"), { target: { value: "30" } });
    await user.click(screen.getByRole("button", { name: "Save Changes" }));

    expect(screen.queryByText(/Invalid input/)).not.toBeInTheDocument();
    expect(mockTagUpdateCall).toHaveBeenCalledWith(
      "sk-test",
      expect.objectContaining({ description: null, rpm_limit: 30 }),
    );
  });

  it("still saves when an invalid limit was left behind in a collapsed budget section", async () => {
    const { user } = await renderEditor();
    const toggle = () => screen.getByRole("button", { name: /Budget & Rate Limits/ });

    await user.click(toggle());
    fireEvent.change(await screen.findByLabelText("RPM Limit"), { target: { value: "1.5" } });
    await user.click(toggle());
    await user.click(screen.getByRole("button", { name: "Save Changes" }));

    const expected = { name: "prod-tag", rpm_limit: undefined, tpm_limit: undefined, max_budget: undefined };
    expect(mockTagUpdateCall).toHaveBeenCalledWith("sk-test", expect.objectContaining(expected));
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
