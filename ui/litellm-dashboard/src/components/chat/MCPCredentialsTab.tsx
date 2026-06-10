"use client";

/**
 * MCPCredentialsTab
 *
 * Shows all OAuth2 MCP connections the calling user has stored.
 * Lives in the Chat sidebar's "Credentials" tab.
 */

import React, { useCallback, useEffect, useState } from "react";
import { Spin } from "antd";
import { Trans, useTranslation } from "react-i18next";
import type { TFunction } from "i18next";
import MessageManager from "@/components/molecules/message_manager";
import { DeleteOutlined, LinkOutlined } from "@ant-design/icons";
import { Badge, Table, TableBody, TableCell, TableHead, TableHeaderCell, TableRow } from "@tremor/react";
import { deleteMCPOAuthUserCredential, listMCPUserCredentials, MCPUserCredentialListItem } from "../networking";

interface Props {
  accessToken: string;
}

function relativeTime(isoString: string | null | undefined, t: TFunction): string {
  if (!isoString) return "";
  try {
    const date = new Date(isoString);
    const diffMs = Date.now() - date.getTime();
    const diffSec = Math.floor(diffMs / 1000);
    if (diffSec < 60) return t("chat.mCPCredentialsTab.justNow");
    const diffMin = Math.floor(diffSec / 60);
    if (diffMin < 60) return t("chat.mCPCredentialsTab.minutesAgo", { count: diffMin });
    const diffHr = Math.floor(diffMin / 60);
    if (diffHr < 24) return t("chat.mCPCredentialsTab.hoursAgo", { count: diffHr });
    return t("chat.mCPCredentialsTab.daysAgo", { count: Math.floor(diffHr / 24) });
  } catch {
    return "";
  }
}

function expiryLabel(isoString: string | null | undefined, t: TFunction): string {
  if (!isoString) return t("chat.mCPCredentialsTab.doesNotExpire");
  try {
    const exp = new Date(isoString);
    const diffMs = exp.getTime() - Date.now();
    if (diffMs <= 0) return t("chat.mCPCredentialsTab.expired");
    const diffSec = Math.floor(diffMs / 1000);
    const diffMin = Math.floor(diffSec / 60);
    const diffHr = Math.floor(diffMin / 60);
    const diffDay = Math.floor(diffHr / 24);
    if (diffDay > 0) return t("chat.mCPCredentialsTab.expiresInDays", { count: diffDay });
    if (diffHr > 0) return t("chat.mCPCredentialsTab.expiresInHours", { count: diffHr });
    return t("chat.mCPCredentialsTab.expiresInMinutes", { count: diffMin });
  } catch {
    return "";
  }
}

const MCPCredentialsTab: React.FC<Props> = ({ accessToken }) => {
  const { t } = useTranslation();
  const [credentials, setCredentials] = useState<MCPUserCredentialListItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [revoking, setRevoking] = useState<Set<string>>(new Set());

  const load = useCallback(() => {
    setLoading(true);
    listMCPUserCredentials(accessToken)
      .then(setCredentials)
      .catch(() => setCredentials([]))
      .finally(() => setLoading(false));
  }, [accessToken]);

  useEffect(() => {
    load();
  }, [load]);

  const handleRevoke = async (serverId: string) => {
    setRevoking((prev) => new Set(prev).add(serverId));
    try {
      await deleteMCPOAuthUserCredential(accessToken, serverId);
      setCredentials((prev) => prev.filter((c) => c.server_id !== serverId));
    } catch {
      MessageManager.error(t("chat.mCPCredentialsTab.revokeError"));
    } finally {
      setRevoking((prev) => {
        const n = new Set(prev);
        n.delete(serverId);
        return n;
      });
    }
  };

  const displayName = (c: MCPUserCredentialListItem) => c.alias || c.server_name || c.server_id;

  return (
    <div className="w-full">
      {/* Header */}
      <div className="mb-4">
        <h2 className="text-base font-semibold text-gray-900 mb-0.5">{t("chat.mCPCredentialsTab.appCredentials")}</h2>
        <p className="text-sm text-gray-500 m-0">{t("chat.mCPCredentialsTab.oauthConnectionsDesc")}</p>
      </div>

      {loading ? (
        <div className="flex justify-center py-12">
          <Spin />
        </div>
      ) : credentials.length === 0 ? (
        <div className="text-center text-gray-400 text-sm py-12 border border-dashed border-gray-200 rounded-lg">
          <LinkOutlined className="text-2xl mb-3 block text-gray-300" />
          {t("chat.mCPCredentialsTab.noConnectionsYet")}
          <br />
          <Trans i18nKey="chat.mCPCredentialsTab.goToAppsConnect" components={{ strong: <strong /> }} />
        </div>
      ) : (
        <div className="rounded-lg border border-gray-200 overflow-hidden">
          <Table>
            <TableHead>
              <TableRow>
                <TableHeaderCell className="text-xs font-medium text-gray-500 py-2 px-4">
                  {t("chat.mCPCredentialsTab.colApp")}
                </TableHeaderCell>
                <TableHeaderCell className="text-xs font-medium text-gray-500 py-2 px-4">
                  {t("chat.mCPCredentialsTab.colConnected")}
                </TableHeaderCell>
                <TableHeaderCell className="text-xs font-medium text-gray-500 py-2 px-4">
                  {t("common.status")}
                </TableHeaderCell>
                <TableHeaderCell className="text-xs font-medium text-gray-500 py-2 px-4 text-right">
                  {t("common.actions")}
                </TableHeaderCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {credentials.map((cred) => {
                const name = displayName(cred);
                const isRevoking = revoking.has(cred.server_id);
                const exp = expiryLabel(cred.expires_at, t);
                const connected = relativeTime(cred.connected_at, t);
                const isExpired = exp === t("chat.mCPCredentialsTab.expired");

                return (
                  <TableRow key={cred.server_id} className="h-10 hover:bg-gray-50">
                    <TableCell className="py-2 px-4">
                      <span className="text-sm font-medium text-gray-900">{name}</span>
                    </TableCell>
                    <TableCell className="py-2 px-4">
                      <span className="text-sm text-gray-500">{connected || "—"}</span>
                    </TableCell>
                    <TableCell className="py-2 px-4">
                      <Badge color={isExpired ? "red" : "green"} size="xs">
                        {exp}
                      </Badge>
                    </TableCell>
                    <TableCell className="py-2 px-4 text-right">
                      <button
                        onClick={() => handleRevoke(cred.server_id)}
                        disabled={isRevoking}
                        title={t("chat.mCPCredentialsTab.revokeConnection")}
                        className={`inline-flex items-center justify-center rounded-md border border-gray-200 px-2 py-1 text-gray-400 hover:text-red-500 hover:border-red-200 transition-colors ${isRevoking ? "opacity-50 cursor-not-allowed" : "cursor-pointer"}`}
                        style={{ background: "none" }}
                      >
                        {isRevoking ? <Spin size="small" /> : <DeleteOutlined className="text-sm" />}
                      </button>
                    </TableCell>
                  </TableRow>
                );
              })}
            </TableBody>
          </Table>
        </div>
      )}
    </div>
  );
};

export default MCPCredentialsTab;
