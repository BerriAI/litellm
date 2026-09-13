import { builtinEnvironments, type Environment } from "vitest/environments";

const env: Environment = {
  name: "jsdom-fetch",
  transformMode: "web",
  async setup(global, options) {
    const nativeAbortController = global.AbortController;
    const nativeAbortSignal = global.AbortSignal;
    const { teardown } = await builtinEnvironments.jsdom.setup(global, options);
    Object.defineProperty(global, "AbortController", {
      configurable: true,
      writable: true,
      value: nativeAbortController,
    });
    Object.defineProperty(global, "AbortSignal", {
      configurable: true,
      writable: true,
      value: nativeAbortSignal,
    });
    return { teardown };
  },
};

export default env;
