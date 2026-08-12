import { screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { renderWithProviders } from "../../../../../tests/test-utils";
import SSOSettingsLoadingSkeleton from "./SSOSettingsLoadingSkeleton";

describe("SSOSettingsLoadingSkeleton", () => {
  it("should render", () => {
    renderWithProviders(<SSOSettingsLoadingSkeleton />);

    expect(screen.getByRole("heading", { name: "SSO Configuration" })).toBeInTheDocument();
  });

  it("should explain which configuration is loading", () => {
    renderWithProviders(<SSOSettingsLoadingSkeleton />);

    expect(screen.getByText("Manage Single Sign-On authentication settings")).toBeInTheDocument();
  });
});
