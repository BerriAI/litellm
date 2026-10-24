"use client";

import React from "react";

import { useKeys } from "@/app/(dashboard)/hooks/keys/useKeys";
import { Card, CardContent, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { formatNumberWithCommas } from "@/utils/dataUtils";
import { keyDetailHref } from "@/utils/entityLinks";

export const TAG_KEYS_PAGE_SIZE = 100;

interface TagKeysSectionProps {
  tagName: string;
}

const TagKeysSection: React.FC<TagKeysSectionProps> = ({ tagName }) => {
  const { data, isLoading, isError } = useKeys(1, TAG_KEYS_PAGE_SIZE, { tag: tagName });
  const keys = data?.keys ?? [];
  const totalCount = data?.total_count ?? 0;

  const renderBody = () => {
    if (isLoading) return <p className="mt-4 text-sm text-muted-foreground">Loading virtual keys...</p>;
    if (isError) return <p className="mt-4 text-sm text-destructive">Could not load the virtual keys for this tag</p>;
    if (keys.length === 0) return <p className="mt-4 text-sm text-muted-foreground">No virtual keys use this tag</p>;
    return (
      <>
        {totalCount > keys.length && (
          <p className="mt-4 text-sm text-muted-foreground">
            Showing the {keys.length} most recently created of {totalCount} keys
          </p>
        )}
        <Table className="mt-4">
          <TableHeader>
            <TableRow>
              <TableHead>Key</TableHead>
              <TableHead>Team</TableHead>
              <TableHead className="text-right">Spend (USD)</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {keys.map((key) => (
              <TableRow key={key.token}>
                <TableCell className="font-medium">
                  <a href={keyDetailHref(key.token)} className="text-info hover:underline">
                    {key.key_alias || key.key_name}
                  </a>
                </TableCell>
                <TableCell>{key.team_id ?? "-"}</TableCell>
                <TableCell className="text-right">{formatNumberWithCommas(key.spend, 4)}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </>
    );
  };

  return (
    <Card>
      <CardContent>
        <CardTitle>Virtual Keys</CardTitle>
        {renderBody()}
      </CardContent>
    </Card>
  );
};

export default TagKeysSection;
