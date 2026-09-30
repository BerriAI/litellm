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

const toggle = () => screen.getByRole("button", { name: /Switch to (light|dark) mode/ });

beforeEach(() => {
  localStorage.clear();
  document.documentElement.classList.remove("dark", "light");
});

afterAll(() => {
  document.documentElement.classList.remove("dark", "light");
});

describe("ThemeToggle", () => {
  it("switches to dark on a single click, with no menu in between", async () => {
    renderToggle();

    await userEvent.click(toggle());

    expect(screen.queryByRole("menu")).not.toBeInTheDocument();
    expect(document.documentElement).toHaveClass("dark");
    expect(localStorage.getItem("theme")).toBe("dark");
  });

  it("switches back to light on the next click", async () => {
    renderToggle();
    await userEvent.click(toggle());

    await userEvent.click(toggle());

    expect(document.documentElement).not.toHaveClass("dark");
    expect(localStorage.getItem("theme")).toBe("light");
  });

  it("names the mode the click will switch to, so the button says what it does", async () => {
    renderToggle();
    expect(screen.getByRole("button", { name: "Switch to dark mode (beta)" })).toBeInTheDocument();

    await userEvent.click(toggle());

    expect(await screen.findByRole("button", { name: "Switch to light mode" })).toBeInTheDocument();
  });

  it("leaves a stored system preference following the OS until the user clicks", () => {
    localStorage.setItem("theme", "system");

    renderToggle();

    expect(localStorage.getItem("theme")).toBe("system");
  });

  it("keeps the beta marker out of the toolbar label once dark is on", async () => {
    renderToggle();

    await userEvent.click(toggle());

    expect(toggle()).not.toHaveTextContent("Beta");
    expect(toggle()).toHaveAccessibleName("Switch to light mode");
  });
});
