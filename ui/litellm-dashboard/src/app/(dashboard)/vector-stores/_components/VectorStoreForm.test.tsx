import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { CredentialItem, vectorStoreCreateCall } from "@/components/networking";
import { Providers, providerLogoMap } from "@/components/provider_info_helpers";
import { VectorStoreProviders } from "@/components/vector_store_providers";
import VectorStoreForm from "./VectorStoreForm";

vi.mock("@/components/networking");

vi.mock("@/components/molecules/notifications_manager", () => ({
  __esModule: true,
  default: {
    success: vi.fn(),
    fromBackend: vi.fn(),
  },
}));

const renderForm = (onCancel: () => void = vi.fn()) =>
  render(
    <VectorStoreForm
      isVisible={true}
      onCancel={onCancel}
      onSuccess={vi.fn()}
      accessToken="test-token"
      credentials={[] as CredentialItem[]}
    />,
  );

describe("VectorStoreForm", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should render the form when visible", () => {
    renderForm();

    expect(screen.getByText("Add New Vector Store")).toBeInTheDocument();
  });

  it("renders the default provider's bundled logo via the shared Logo component", () => {
    renderForm();

    const logo = screen.getByRole("img", { name: `${VectorStoreProviders.Bedrock} logo` });
    expect(logo).toHaveAttribute("src", providerLogoMap[Providers.Bedrock]);
  });

  it("creates the vector store when Create is clicked on a filled form", async () => {
    const user = userEvent.setup();
    renderForm();

    await user.type(screen.getByLabelText(/Vector Store ID/), "vs-created");
    await user.click(screen.getByRole("button", { name: "Create" }));

    await vi.waitFor(() => expect(vectorStoreCreateCall).toHaveBeenCalledTimes(1));
    expect(vi.mocked(vectorStoreCreateCall).mock.calls[0][1]).toMatchObject({ vector_store_id: "vs-created" });
  });

  it("cancels without creating the vector store when Cancel is clicked on a filled form", async () => {
    const user = userEvent.setup();
    const onCancel = vi.fn();
    renderForm(onCancel);

    await user.type(screen.getByLabelText(/Vector Store ID/), "vs-abandoned");
    await user.click(screen.getByRole("button", { name: "Cancel" }));

    expect(onCancel).toHaveBeenCalledTimes(1);
    expect(vectorStoreCreateCall).not.toHaveBeenCalled();
  });
});
