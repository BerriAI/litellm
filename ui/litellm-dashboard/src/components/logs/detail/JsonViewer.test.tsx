import { render, screen } from "@testing-library/react";
import { ThemeProvider } from "next-themes";
import { darkStyles, defaultStyles } from "react-json-view-lite";
import { describe, expect, it } from "vitest";
import { JsonViewer } from "./JsonViewer";

const renderWithTheme = (theme: "light" | "dark", data: unknown) =>
  render(
    <ThemeProvider attribute="class" defaultTheme={theme} enableSystem={false}>
      <JsonViewer data={data} mode="formatted" />
    </ThemeProvider>,
  );

describe("JsonViewer", () => {
  it("should render a placeholder and no tree when the log entry carries no payload", () => {
    renderWithTheme("light", null);

    expect(screen.getByText("No data")).toBeInTheDocument();
    expect(screen.queryByRole("tree")).not.toBeInTheDocument();
  });

  it("should render the payload as a tree exposing its keys", () => {
    renderWithTheme("light", { model: "claude-opus-4-5", stream: true });

    expect(screen.getByRole("tree")).toBeInTheDocument();
    expect(screen.getByText(/model/)).toBeInTheDocument();
    expect(screen.getByText(/stream/)).toBeInTheDocument();
    expect(screen.queryByText("No data")).not.toBeInTheDocument();
  });

  it("should treat an empty payload as data rather than showing the placeholder", () => {
    renderWithTheme("light", {});

    expect(screen.getByRole("tree")).toBeInTheDocument();
    expect(screen.queryByText("No data")).not.toBeInTheDocument();
  });

  it("should style the tree with the light palette when the dashboard theme is light", () => {
    renderWithTheme("light", { model: "claude-opus-4-5" });

    expect(screen.getByRole("tree")).toHaveClass(...defaultStyles.container.split(" "));
  });

  it("should style the tree with the dark palette when the dashboard theme is dark", () => {
    renderWithTheme("dark", { model: "claude-opus-4-5" });

    const tree = screen.getByRole("tree");
    expect(tree).toHaveClass(...darkStyles.container.split(" "));
    defaultStyles.container
      .split(" ")
      .filter((className) => !darkStyles.container.split(" ").includes(className))
      .forEach((lightOnlyClassName) => expect(tree).not.toHaveClass(lightOnlyClassName));
  });
});
