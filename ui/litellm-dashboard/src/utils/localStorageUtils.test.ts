import { describe, it, expect, vi, beforeEach } from "vitest";
import {
  emitLocalStorageChange,
  LOCAL_STORAGE_EVENT,
  getLocalStorageItem,
  setLocalStorageItem,
  removeLocalStorageItem,
} from "./localStorageUtils";

describe("emitLocalStorageChange", () => {
  it("should dispatch a custom event with the provided key", () => {
    const dispatchEventSpy = vi.spyOn(window, "dispatchEvent");
    const testKey = "test-key";

    emitLocalStorageChange(testKey);

    expect(dispatchEventSpy).toHaveBeenCalledWith(expect.any(CustomEvent));

    const dispatchedEvent = dispatchEventSpy.mock.calls[0][0] as CustomEvent;
    expect(dispatchedEvent.type).toBe(LOCAL_STORAGE_EVENT);
    expect(dispatchedEvent.detail).toEqual({ key: testKey });

    dispatchEventSpy.mockRestore();
  });
});

describe("getLocalStorageItem", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should return null when window is undefined", () => {
    const originalWindow = global.window;
    // @ts-ignore
    delete global.window;

    expect(getLocalStorageItem("test-key")).toBeNull();

    global.window = originalWindow;
  });

  it("should return the stored value when it exists", () => {
    const getItemSpy = vi.spyOn(Storage.prototype, "getItem").mockReturnValue("test-value");

    const result = getLocalStorageItem("test-key");

    expect(result).toBe("test-value");
    expect(getItemSpy).toHaveBeenCalledWith("test-key");

    getItemSpy.mockRestore();
  });

  it("should return null and log warning when localStorage throws an error", () => {
    const consoleSpy = vi.spyOn(console, "warn").mockImplementation(() => {});
    const getItemSpy = vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => {
      throw new Error("Storage quota exceeded");
    });

    const result = getLocalStorageItem("test-key");

    expect(result).toBeNull();
    expect(consoleSpy).toHaveBeenCalledWith('Error reading localStorage key "test-key":', expect.any(Error));

    consoleSpy.mockRestore();
    getItemSpy.mockRestore();
  });
});

describe("setLocalStorageItem", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should do nothing when window is undefined", () => {
    const originalWindow = global.window;
    // @ts-ignore
    delete global.window;

    const setItemSpy = vi.spyOn(Storage.prototype, "setItem");

    setLocalStorageItem("test-key", "test-value");

    expect(setItemSpy).not.toHaveBeenCalled();

    global.window = originalWindow;
    setItemSpy.mockRestore();
  });

  it("should set the item in localStorage", () => {
    const setItemSpy = vi.spyOn(Storage.prototype, "setItem");

    setLocalStorageItem("test-key", "test-value");

    expect(setItemSpy).toHaveBeenCalledWith("test-key", "test-value");

    setItemSpy.mockRestore();
  });

  it("should log warning when localStorage throws an error", () => {
    const consoleSpy = vi.spyOn(console, "warn").mockImplementation(() => {});
    const setItemSpy = vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
      throw new Error("Storage quota exceeded");
    });

    setLocalStorageItem("test-key", "test-value");

    expect(consoleSpy).toHaveBeenCalledWith('Error setting localStorage key "test-key":', expect.any(Error));

    consoleSpy.mockRestore();
    setItemSpy.mockRestore();
  });
});

describe("removeLocalStorageItem", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should do nothing when window is undefined", () => {
    const originalWindow = global.window;
    // @ts-ignore
    delete global.window;

    const removeItemSpy = vi.spyOn(Storage.prototype, "removeItem");

    removeLocalStorageItem("test-key");

    expect(removeItemSpy).not.toHaveBeenCalled();

    global.window = originalWindow;
    removeItemSpy.mockRestore();
  });

  it("should remove the item from localStorage", () => {
    const removeItemSpy = vi.spyOn(Storage.prototype, "removeItem");

    removeLocalStorageItem("test-key");

    expect(removeItemSpy).toHaveBeenCalledWith("test-key");

    removeItemSpy.mockRestore();
  });

  it("should log warning when localStorage throws an error", () => {
    const consoleSpy = vi.spyOn(console, "warn").mockImplementation(() => {});
    const removeItemSpy = vi.spyOn(Storage.prototype, "removeItem").mockImplementation(() => {
      throw new Error("Storage operation failed");
    });

    removeLocalStorageItem("test-key");

    expect(consoleSpy).toHaveBeenCalledWith('Error removing localStorage key "test-key":', expect.any(Error));

    consoleSpy.mockRestore();
    removeItemSpy.mockRestore();
  });
});
