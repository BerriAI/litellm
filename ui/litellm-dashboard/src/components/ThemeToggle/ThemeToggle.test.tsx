import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ThemeProvider } from "next-themes";
import { afterAll, beforeEach, describe, expect, it } from "vitest";
import ThemeToggle from "./ThemeToggle";

const renderToggle = () =>
  render(
    <ThemeProvider attribute="class" defaultTheme="light" enableSystem disableTransitionOnChange>
      <ThemeToggle />
    </ThemeProvider>,
  );

const openMenu = async () => {
  await userEvent.click(screen.getByRole("button", { name: "Theme" }));
  await screen.findByRole("menu");
};

const pick = async (label: string) => userEvent.click(screen.getByRole("menuitemradio", { name: label }));

beforeEach(() => {
  localStorage.clear();
  document.documentElement.classList.remove("dark", "light");
});

afterAll(() => {
  document.documentElement.classList.remove("dark", "light");
});

describe("ThemeToggle", () => {
  it("starts on light rather than following the system preference", async () => {
    renderToggle();
    await openMenu();

    expect(screen.getByRole("menuitemradio", { name: "Light" })).toBeChecked();
    expect(screen.getByRole("menuitemradio", { name: "Dark" })).not.toBeChecked();
    expect(screen.getByRole("menuitemradio", { name: "System" })).not.toBeChecked();
  });

  it("puts the dark class on the document and remembers the choice", async () => {
    renderToggle();
    await openMenu();

    await pick("Dark");

    expect(document.documentElement).toHaveClass("dark");
    expect(localStorage.getItem("theme")).toBe("dark");
  });

  it("hands control back to the system preference when asked", async () => {
    renderToggle();
    await openMenu();
    await pick("Dark");

    await pick("System");

    expect(localStorage.getItem("theme")).toBe("system");
    expect(document.documentElement).not.toHaveClass("dark");
  });

  it("flags dark mode as experimental, and only while it is on", async () => {
    renderToggle();
    await openMenu();
    expect(screen.queryByText("Experimental")).not.toBeInTheDocument();

    await pick("Dark");
    expect(screen.getByText("Experimental")).toBeInTheDocument();

    await pick("Light");
    expect(screen.queryByText("Experimental")).not.toBeInTheDocument();
  });
});
