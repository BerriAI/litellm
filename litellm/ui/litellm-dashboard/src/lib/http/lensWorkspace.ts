export type LensWorkspaceAvailability = "served" | "unsupported" | "unavailable";

/** Whether the Lens connected to this gateway serves the workspace the dashboard frames. */
export const lensWorkspaceAvailability = async (
  url: string,
  signal: AbortSignal,
): Promise<LensWorkspaceAvailability> => {
  try {
    const response = await fetch(url, { signal });
    if (response.ok) return "served";
    return response.status === 404 ? "unsupported" : "unavailable";
  } catch {
    return "unavailable";
  }
};
