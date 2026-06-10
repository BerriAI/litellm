import React, { useState } from "react";
import { Icon } from "@tremor/react";
import { EyeIcon } from "@heroicons/react/outline";
import { Tooltip, Tag, Popover, Spin } from "antd";
import { useTranslation, Trans } from "react-i18next";
import { PolicyAttachment } from "./types";
import { estimateAttachmentImpactCall } from "../networking";

const ImpactPopover: React.FC<{ attachment: PolicyAttachment; accessToken: string | null }> = ({
  attachment,
  accessToken,
}) => {
  const { t } = useTranslation();
  const [impact, setImpact] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [loaded, setLoaded] = useState(false);

  const loadImpact = async () => {
    if (loaded || loading || !accessToken) return;
    setLoading(true);
    try {
      const data = await estimateAttachmentImpactCall(accessToken, {
        policy_name: attachment.policy_name,
        scope: attachment.scope,
        teams: attachment.teams,
        keys: attachment.keys,
        models: attachment.models,
        tags: attachment.tags,
      });
      setImpact(data);
      setLoaded(true);
    } catch (error) {
      console.error("Failed to load impact:", error);
    } finally {
      setLoading(false);
    }
  };

  const content = loading ? (
    <div className="p-2 text-center">
      <Spin size="small" /> {t("common.loading")}
    </div>
  ) : impact ? (
    <div className="text-xs" style={{ maxWidth: 280 }}>
      {impact.affected_keys_count === -1 ? (
        <p className="font-medium text-amber-600">{t("policies.impactPopover.globalScope")}</p>
      ) : (
        <>
          <p className="mb-1">
            <Trans
              i18nKey="policies.impactPopover.keysTeamsAffected"
              values={{
                keysCount: impact.affected_keys_count,
                keysWord: t("policies.impactPopover.keyWord", { count: impact.affected_keys_count }),
                teamsCount: impact.affected_teams_count,
                teamsWord: t("policies.impactPopover.teamWord", { count: impact.affected_teams_count }),
              }}
              components={[<strong key="keys" />, <strong key="teams" />]}
            />
          </p>
          {impact.sample_keys.length > 0 && (
            <div className="mb-1">
              <span className="text-gray-500">{t("policies.impactPopover.keysLabel")} </span>
              {impact.sample_keys.map((k: string) => (
                <Tag key={k} style={{ fontSize: 10, margin: 1 }}>
                  {k}
                </Tag>
              ))}
            </div>
          )}
          {impact.sample_teams.length > 0 && (
            <div>
              <span className="text-gray-500">{t("policies.impactPopover.teamsLabel")} </span>
              {impact.sample_teams.map((teamName: string) => (
                <Tag key={teamName} style={{ fontSize: 10, margin: 1 }}>
                  {teamName}
                </Tag>
              ))}
            </div>
          )}
          {impact.affected_keys_count === 0 && impact.affected_teams_count === 0 && (
            <p className="text-gray-400">{t("policies.impactPopover.noKeysOrTeams")}</p>
          )}
        </>
      )}
    </div>
  ) : (
    <p className="text-xs text-gray-400">{t("policies.impactPopover.clickToLoad")}</p>
  );

  return (
    <Popover
      content={content}
      title={t("policies.impactPopover.blastRadius")}
      trigger="click"
      onOpenChange={(open) => {
        if (open) loadImpact();
      }}
    >
      <Tooltip title={t("policies.impactPopover.viewBlastRadius")}>
        <Icon icon={EyeIcon} size="sm" className="cursor-pointer hover:text-blue-500" />
      </Tooltip>
    </Popover>
  );
};

export default ImpactPopover;
