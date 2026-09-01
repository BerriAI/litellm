import React, { PropsWithChildren } from "react";
import { render, RenderOptions, screen, waitFor } from "@testing-library/react";
import type userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { NuqsTestingAdapter, OnUrlUpdateFunction } from "nuqs/adapters/testing";
import { expect } from "vitest";

// Create a client for testing
export const testQueryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: false,
      gcTime: Infinity,
      staleTime: Infinity,
      refetchOnWindowFocus: false,
      refetchOnReconnect: false,
      refetchOnMount: false,
    },
    mutations: {
      retry: false,
    },
  },
});

interface ProviderOptions {
  searchParams?: string | Record<string, string> | URLSearchParams;
  onUrlUpdate?: OnUrlUpdateFunction;
}

export const renderWithProviders = (ui: React.ReactElement, options?: RenderOptions & ProviderOptions) => {
  const { searchParams, onUrlUpdate, ...renderOptions } = options ?? {};
  const Providers: React.FC<PropsWithChildren> = ({ children }) => (
    <NuqsTestingAdapter searchParams={searchParams} onUrlUpdate={onUrlUpdate} hasMemory>
      <QueryClientProvider client={testQueryClient}>{children}</QueryClientProvider>
    </NuqsTestingAdapter>
  );
  return render(ui, { wrapper: Providers, ...renderOptions });
};

const pointerBlocked = (element: HTMLElement): boolean => {
  for (let node: HTMLElement | null = element; node !== null; node = node.parentElement) {
    if (node.style.pointerEvents === "none") return true;
  }
  return false;
};

/**
 * Opens a Base UI popup and picks an entry by its accessible name.
 *
 * Querying the entry by text or by a title attribute matches the moment the node exists, which is
 * one render before the popup finishes entering. Until then the positioner still carries
 * `pointer-events: none` and user-event refuses to click, so that shape is a race a fast machine
 * loses. The role query only matches once the popup is open to the accessibility tree, which is
 * what makes this wait correct rather than lucky.
 */
export const chooseSelectOption = async (
  user: Pick<ReturnType<typeof userEvent.setup>, "click">,
  trigger: HTMLElement,
  optionName: string | RegExp,
  role: "option" | "menuitem" | "menuitemradio" = "option",
) => {
  await user.click(trigger);
  const option = await screen.findByRole(role, { name: optionName });
  await waitFor(() => expect(pointerBlocked(option)).toBe(false));
  await user.click(option);
};

export * from "@testing-library/react";
