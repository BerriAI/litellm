import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import LabeledField from "./LabeledField";

vi.mock("next/navigation", () => ({ useRouter: () => ({ push: vi.fn() }) }));

describe("LabeledField", () => {
  it("should render the label and value", () => {
    render(<LabeledField label="User Email" value="test@example.com" />);
    expect(screen.getByText("User Email")).toBeInTheDocument();
    expect(screen.getByText("test@example.com")).toBeInTheDocument();
  });

  it("should render the icon when provided", () => {
    render(<LabeledField label="Name" value="Alice" icon={<span data-testid="test-icon" />} />);
    expect(screen.getByTestId("test-icon")).toBeInTheDocument();
  });

  it("should show '-' when value is empty", () => {
    render(<LabeledField label="User ID" value="" />);
    expect(screen.getByText("-")).toBeInTheDocument();
  });

  it("should show 'Default Proxy Admin' tag when value is default_user_id and defaultUserIdCheck is true", () => {
    render(<LabeledField label="User ID" value="default_user_id" copyable defaultUserIdCheck />);
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
    expect(screen.queryByRole("button", { name: "Copy User ID" })).not.toBeInTheDocument();
  });

  it("should not be copyable when value is default_user_id and defaultUserIdCheck is true", () => {
    render(<LabeledField label="User ID" value="default_user_id" copyable defaultUserIdCheck />);
    expect(screen.queryByRole("button", { name: "Copy User ID" })).not.toBeInTheDocument();
  });

  it("should not be copyable when copyable is false", () => {
    render(<LabeledField label="User ID" value="user-123" />);
    expect(screen.queryByRole("button", { name: "Copy User ID" })).not.toBeInTheDocument();
  });

  it("should be copyable when copyable is true and value is present", () => {
    render(<LabeledField label="User ID" value="user-123" copyable />);
    expect(screen.getByRole("button", { name: "Copy User ID" })).toBeInTheDocument();
  });

  it("should render the value as a link when href is provided", () => {
    render(<LabeledField label="Team" value="my-team" href="/ui/teams?team=t1" />);
    expect(screen.getByRole("link", { name: "my-team" })).toHaveAttribute("href", "/ui/teams?team=t1");
  });

  it("should keep the copy button next to a linked value", () => {
    render(<LabeledField label="Created By" value="alice" href="/ui/users?user=u1" copyable />);
    expect(screen.getByRole("link", { name: "alice" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Copy Created By" })).toBeInTheDocument();
  });

  it("should not link an empty value even when href is provided", () => {
    render(<LabeledField label="Team" value="" href="/ui/teams?team=t1" />);
    expect(screen.queryByRole("link")).not.toBeInTheDocument();
    expect(screen.getByText("-")).toBeInTheDocument();
  });

  it("should not link the Default Proxy Admin tag", () => {
    render(
      <LabeledField
        label="Created By"
        value="default_user_id"
        href="/ui/users?user=default_user_id"
        defaultUserIdCheck
      />,
    );
    expect(screen.getByText("Default Proxy Admin")).toBeInTheDocument();
    expect(screen.queryByRole("link")).not.toBeInTheDocument();
  });
});
