import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { CredentialItem } from "@/components/networking";
import { Providers, providerLogoMap } from "@/components/provider_info_helpers";
import { VectorStoreProviders } from "@/components/vector_store_providers";
import VectorStoreForm from "./VectorStoreForm";

vi.mock("@/components/networking");

const renderForm = () =>
  render(
    <VectorStoreForm
      isVisible={true}
      onCancel={vi.fn()}
      onSuccess={vi.fn()}
      accessToken="test-token"
      credentials={[] as CredentialItem[]}
    />,
  );

describe("VectorStoreForm", () => {
  it("should render the form when visible", () => {
    renderForm();

    expect(screen.getByText("Add New Vector Store")).toBeInTheDocument();
  });

  it("renders the default provider's bundled logo via the shared Logo component", () => {
    renderForm();

    const logo = screen.getByRole("img", { name: `${VectorStoreProviders.Bedrock} logo` });
    expect(logo.getAttribute("src")).toBe(providerLogoMap[Providers.Bedrock]);
  });
});
