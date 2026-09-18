// Query keys factory

type ListParams = {
  page?: number;
  limit?: number;
  filters?: Record<string, string | number>;
};

/**
 * Generates a query keys factory for a given resource.
 *
 * @param resource - The name of the resource (e.g., "books", "users", "keys")
 * @returns An object with query key generators following the standard pattern
 *
 * @example
 * ```ts
 * const bookKeys = createQueryKeys("books");
 * // bookKeys.all -> ["books"]
 * // bookKeys.lists() -> ["books", "list"]
 * // bookKeys.list({ page: 1 }) -> ["books", "list", { params: { page: 1 } }]
 * // bookKeys.details() -> ["books", "detail"]
 * // bookKeys.detail("123") -> ["books", "detail", "123"]
 * ```
 */
export function createQueryKeys<T extends string>(resource: T) {
  const all = [resource] as const;

  return {
    all,
    lists: () => [...all, "list"] as const,
    list: (params?: ListParams) => [...all, "list", { params }] as const,
    details: () => [...all, "detail"] as const,
    detail: (uid: string) => [...all, "detail", uid] as const,
  };
}
