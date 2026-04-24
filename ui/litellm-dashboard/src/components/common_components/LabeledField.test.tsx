import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import LabeledField from "./LabeledField";

describe("LabeledField", () => {
  it("should render the label and value", () => {
    render(<LabeledField label="User Email" value="test@example.com" />);
    expect(screen.getByText("User Email")).toBeInTheDocument();
    expect(screen.getByText("test@example.com")).toBeInTheDocument();
  });

  it("should render the icon when provided", () => {
    render(
      <LabeledField label="Name" value="Alice" icon={<span data-testid="test-icon" />} />,
    );
    expect(screen.getByTestId("test-icon")).toBeInTheDocument();
  });

  it("should show '-' when value is empty", () => {
    render(<LabeledField label="User ID" value="" />);
    expect(screen.getByText("-")).toBeInTheDocument();
  });

  it("should show 'Default Proxy Admin' tag when value is default_user_id and defaultUserIdCheck is true", () => {
    render(
      <LabeledField label="User ID" value="default_user_id" copyable defaultUserIdCheck />,
    );
    expect(screen.getByText("Default Proxy Admin")).toBeInTheDocument();
    expect(screen.queryByText("default_user_id")).not.toBeInTheDocument();
  });

  it("should show raw value when value is default_user_id but defaultUserIdCheck is false", () => {
    render(<LabeledField label="User ID" value="default_user_id" />);
    expect(screen.getByText("default_user_id")).toBeInTheDocument();
    expect(screen.queryByText("Default Proxy Admin")).not.toBeInTheDocument();
  });

  it("should not be copyable when value is empty", () => {
    render(<LabeledField label="User ID" value="" copyable />);
    expect(
      screen.queryByRole("button", { name: /copy user id/i }),
    ).not.toBeInTheDocument();
  });

  it("should not be copyable when value is default_user_id and defaultUserIdCheck is true", () => {
    render(
      <LabeledField label="User ID" value="default_user_id" copyable defaultUserIdCheck />,
    );
    expect(
      screen.queryByRole("button", { name: /copy user id/i }),
    ).not.toBeInTheDocument();
  });

  it("should be copyable when copyable is true and value is present", () => {
    render(<LabeledField label="User ID" value="user-123" copyable />);
    expect(
      screen.getByRole("button", { name: /copy user id/i }),
    ).toBeInTheDocument();
  });
});
