import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ThemeProvider } from "next-themes";
import { afterAll, beforeEach, describe, expect, it } from "vitest";
import ThemeToggle from "./ThemeToggle";

const renderToggle = () =>
  render(
    <ThemeProvider attribute="class" defaultTheme="light" enableSystem>
      <ThemeToggle />
    </ThemeProvider>,
  );

beforeEach(() => {
  localStorage.clear();
  document.documentElement.classList.remove("dark", "light");
});

afterAll(() => {
  document.documentElement.classList.remove("dark", "light");
});

describe("ThemeToggle", () => {
  it("marks only the active theme as chosen", () => {
    renderToggle();

    expect(screen.getByRole("radio", { name: "Light" })).toBeChecked();
    expect(screen.getByRole("radio", { name: "Dark" })).not.toBeChecked();
    expect(screen.getByRole("radio", { name: "System" })).not.toBeChecked();
  });

  it("puts the dark class on the document and remembers the choice", async () => {
    renderToggle();

    await userEvent.click(screen.getByRole("radio", { name: "Dark" }));

    expect(document.documentElement).toHaveClass("dark");
    expect(localStorage.getItem("theme")).toBe("dark");
  });

  it("hands control back to the system preference", async () => {
    renderToggle();
    await userEvent.click(screen.getByRole("radio", { name: "Dark" }));

    await userEvent.click(screen.getByRole("radio", { name: "System" }));

    expect(localStorage.getItem("theme")).toBe("system");
    expect(document.documentElement).not.toHaveClass("dark");
  });
});
