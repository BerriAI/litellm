import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import UsageTags from "./UsageTags";

describe("model usage tags", () => {
  it("should render tags used with the model", () => {
    render(<UsageTags tags={["production"]} loading={false} failed={false} />);
    expect(screen.getByText("production")).toBeInTheDocument();
  });
  it("should show five tags and let the user expand and collapse the rest", async () => {
    const user = userEvent.setup();
    render(<UsageTags tags={["one", "two", "three", "four", "five", "six", "seven"]} loading={false} failed={false} />);
    expect(screen.queryByText("six")).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Show 2 more usage tags" }));
    expect(screen.getByText("six")).toBeInTheDocument();
    expect(screen.getByText("seven")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Show fewer usage tags" }));
    expect(screen.queryByText("six")).not.toBeInTheDocument();
  });
});
