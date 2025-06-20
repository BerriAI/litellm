import { useState, useMemo, useCallback } from "react";

const DEFAULT_PAGE_SIZE = 30;

type PaginationState = {
  start: number;
  end: number;
  pages: number;
  total: number;
  setPage: React.Dispatch<React.SetStateAction<number>>;
  scrollContainer?: HTMLDivElement | null;
  canGoBack: boolean;
  canGoForward: boolean;
  page: number;
  goBack(): void;
  goForward(): void;
};

type UsePaginationProps = {
  total: number;
  scrollContainer?: HTMLDivElement | null;
  pageSize?: number;
};

export function usePagination({
  pageSize = DEFAULT_PAGE_SIZE,
  ...props
}: UsePaginationProps): PaginationState {
  const pages = Math.ceil(props.total / pageSize);
  const [page, setPage] = useState(1);
  const resolvedPage = Math.max(Math.min(page, pages), 1);
  const start = Math.min((resolvedPage - 1) * pageSize + 1, props.total);
  const end = Math.min(start + pageSize - 1, props.total);

  const canGoBack = useMemo(() => resolvedPage > 1, [resolvedPage]);
  const canGoForward = useMemo(
    () => resolvedPage < pages,
    [resolvedPage, pages],
  );

  const goBack = useCallback(() => {
    if (resolvedPage === 1) return;

    setPage((currentPage) => Math.max(currentPage - 1, 1));
    props.scrollContainer?.scrollTo({ top: 0, left: 0 });
  }, [props.scrollContainer, resolvedPage]);

  const goForward = useCallback(() => {
    if (resolvedPage >= pages) return;

    setPage((currentPage) => Math.min(currentPage + 1, pages));
    props.scrollContainer?.scrollTo({ top: 0, left: 0 });
  }, [props.scrollContainer, resolvedPage, pages]);

  const state = useMemo(() => {
    const _state: PaginationState = {
      start,
      end,
      pages,
      page: resolvedPage,
      total: props.total,
      scrollContainer: props.scrollContainer,
      setPage,
      canGoBack,
      canGoForward,
      goBack,
      goForward,
    };

    return _state;
  }, [
    start,
    end,
    pages,
    props.total,
    resolvedPage,
    props.scrollContainer,
    setPage,
    canGoBack,
    canGoForward,
    goBack,
    goForward,
  ]);

  return state;
}
