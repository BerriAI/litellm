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
 * Opens a Base UI Select and picks an option by its accessible name.
 *
 * The option is in the DOM one render before the popup finishes entering, and until then its
 * positioner still carries `pointer-events: none`, which user-event refuses to click. Waiting on
 * the option text alone is a race that React 19's flush timing loses.
 */
export const chooseSelectOption = async (
  user: ReturnType<typeof userEvent.setup>,
  trigger: HTMLElement,
  optionName: string | RegExp,
) => {
  await user.click(trigger);
  const option = await screen.findByRole("option", { name: optionName });
  await waitFor(() => expect(pointerBlocked(option)).toBe(false));
  await user.click(option);
};

export * from "@testing-library/react";
