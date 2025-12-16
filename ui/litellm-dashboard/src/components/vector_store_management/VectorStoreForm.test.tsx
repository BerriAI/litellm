import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import VectorStoreForm from "./VectorStoreForm";
import * as networking from "../networking";
import { CredentialItem } from "../networking";

vi.mock("../networking");

describe("VectorStoreForm", () => {
  it("should render the form when visible", () => {
    const mockOnCancel = vi.fn();
    const mockOnSuccess = vi.fn();
    const mockAccessToken = "test-token";
    const mockCredentials: CredentialItem[] = [];

    render(
      <VectorStoreForm
        isVisible={true}
        onCancel={mockOnCancel}
        onSuccess={mockOnSuccess}
        accessToken={mockAccessToken}
        credentials={mockCredentials}
      />,
    );

    expect(screen.getByText("Add New Vector Store")).toBeInTheDocument();
  });
});
