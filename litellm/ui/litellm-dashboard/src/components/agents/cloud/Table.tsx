import type { ReactNode } from "react";
import { useTranslation } from "react-i18next";

export function CloudTable<T>({
  rows,
  columns,
  rowKey,
}: {
  rows: readonly T[];
  columns: readonly { key: string; heading: string; cell: (row: T) => ReactNode }[];
  rowKey: (row: T) => string;
}) {
  const { t } = useTranslation();
  return (
    <div className="overflow-x-auto rounded-lg border bg-card">
      <table className="w-full text-left text-sm">
        <thead className="border-b bg-muted/40 text-muted-foreground">
          <tr>
            {columns.map((column) => (
              <th className="whitespace-nowrap px-4 py-3 font-medium" key={column.key} scope="col">
                {column.heading}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y">
          {rows.map((row) => (
            <tr key={rowKey(row)} className="hover:bg-muted/20">
              {columns.map((column) => (
                <td key={column.key} className="px-4 py-4 align-top">
                  {column.cell(row)}
                </td>
              ))}
            </tr>
          ))}
          {rows.length === 0 && (
            <tr>
              <td className="p-8 text-center text-muted-foreground" colSpan={columns.length}>
                {t("agents.external.noRows")}
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}
