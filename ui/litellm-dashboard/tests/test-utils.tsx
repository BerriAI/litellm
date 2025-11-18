import React, { PropsWithChildren } from "react";
import { render, RenderOptions } from "@testing-library/react";

const Providers: React.FC<PropsWithChildren> = ({ children }) => {
  // Add future providers here (Theme/Router/QueryClient/etc.)
  return <>{children}</>;
};

export const renderWithProviders = (ui: React.ReactElement, options?: RenderOptions) =>
  render(ui, { wrapper: Providers, ...options });

export * from "@testing-library/react";
