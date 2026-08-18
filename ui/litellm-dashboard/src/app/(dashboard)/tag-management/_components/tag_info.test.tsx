import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { tagInfoCall, tagUpdateCall } from "@/components/networking";

import TagInfoView from "./tag_info";

vi.mock("@/components/networking", () => ({
  tagInfoCall: vi.fn(),
  tagUpdateCall: vi.fn(),
  modelAvailableCall: vi.fn(),
}));

vi.mock("@/components/organisms/create_key_button", () => ({
  fetchUserModels: vi.fn(),
}));

vi.mock("@/components/molecules/notifications_manager", () => ({
  __esModule: true,
  default: {
    success: vi.fn(),
    fromBackend: vi.fn(),
  },
}));

const mockTagInfoCall = vi.mocked(tagInfoCall);

const renderEditor = () =>
  render(<TagInfoView tagId="prod" onClose={vi.fn()} accessToken="sk-test" is_admin={true} editTag={true} />);

describe("TagInfoView editor", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockTagInfoCall.mockResolvedValue({
      prod: {
        name: "prod",
        description: "production traffic",
        models: [],
        created_at: "2024-01-01T00:00:00Z",
        updated_at: "2024-01-02T00:00:00Z",
      },
    });
  });

  it("saves the edited tag when Save Changes is clicked", async () => {
    const user = userEvent.setup();
    renderEditor();

    const description = await screen.findByLabelText("Description");
    await user.clear(description);
    await user.type(description, "staging traffic");
    await user.click(screen.getByRole("button", { name: "Save Changes" }));

    await vi.waitFor(() => expect(tagUpdateCall).toHaveBeenCalledTimes(1));
    expect(vi.mocked(tagUpdateCall).mock.calls[0][1]).toMatchObject({
      name: "prod",
      description: "staging traffic",
    });
  });

  it("leaves the tag untouched when Cancel is clicked", async () => {
    const user = userEvent.setup();
    renderEditor();

    const description = await screen.findByLabelText("Description");
    await user.clear(description);
    await user.type(description, "staging traffic");
    await user.click(screen.getByRole("button", { name: "Cancel" }));

    expect(await screen.findByText("Tag Details")).toBeInTheDocument();
    expect(tagUpdateCall).not.toHaveBeenCalled();
  });
});
