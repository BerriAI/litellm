import React from 'react';
import { QuestionCircleOutlined } from '@ant-design/icons';
import { Alert, Card, Empty, Input, Tag, Tooltip, Typography } from 'antd';
import { useGetKeyModels } from '@/hooks/keys/useGetKeyModels';
import type { KeyModelDisplaySection } from '@/components/networking';

const { Text } = Typography;

/** Card body (search + list) max height — list area scrolls inside flex layout. */
const CARD_BODY_MAX_HEIGHT = 'min(72vh, 580px)';

interface KeyModelListProps {
  key_id: string;
}

const INNER_CARD_CLASS = 'mb-3 last:mb-0';

const SectionBlock: React.FC<{
  section: KeyModelDisplaySection;
  warnNoTeam: boolean;
}> = ({ section, warnNoTeam }) => {
  const tagRow = (
    <div className="flex flex-wrap gap-2">
      {section.models.map((item, index) => (
        <Tag key={`${section.section_kind}-${section.title}-${item}-${index}`}>{item}</Tag>
      ))}
    </div>
  );

  if (section.section_kind === 'access_group') {
    return (
      <Card
        size="small"
        type="inner"
        className={INNER_CARD_CLASS}
        data-section-kind={section.section_kind}
        title={
          <span>
            <Text code>access_group</Text>
            <Text strong className="ml-2">
              {section.title}
            </Text>
          </span>
        }
      >
        {tagRow}
      </Card>
    );
  }

  if (section.section_kind === 'all_proxy_models' || section.section_kind === 'all_team_models') {
    const showNoTeamHint = section.section_kind === 'all_team_models' && warnNoTeam;
    const title = (
      <span className="inline-flex items-center gap-2">
        <Text strong>{section.title}</Text>
        {showNoTeamHint ? (
          <Tooltip
            title='This key uses "All team models" but has no team assigned. Assign a team in key settings so access follows that team model list.'
            placement="topLeft"
          >
            <QuestionCircleOutlined
              className="text-black cursor-help text-base"
              aria-label="Why this matters"
            />
          </Tooltip>
        ) : null}
      </span>
    );
    return (
      <Card
        size="small"
        type="inner"
        className={INNER_CARD_CLASS}
        data-section-kind={section.section_kind}
        title={title}
      >
        {tagRow}
      </Card>
    );
  }

  return (
    <Card
      size="small"
      type="inner"
      className={INNER_CARD_CLASS}
      data-section-kind={section.section_kind}
      title={<Text strong>{section.title}</Text>}
    >
      {tagRow}
    </Card>
  );
};

const KeyModelList: React.FC<KeyModelListProps> = ({ key_id }) => {
  const {
    searchInput,
    setSearchInput,
    defaultModelsQuery,
    searchQuery,
    hasActiveSearch,
    isInitialLoading,
    searchInputLoading,
  } = useGetKeyModels(key_id);

  const defaultData = defaultModelsQuery.data;
  const searchData = searchQuery.data;
  const warnNoTeam =
    (searchData?.all_team_models_without_team ?? defaultData?.all_team_models_without_team) === true;

  const sections = hasActiveSearch ? searchData?.model_display_sections : defaultData?.model_display_sections;
  const truncated = hasActiveSearch ? searchData?.models_truncated : defaultData?.models_truncated;
  const nothingFound =
    hasActiveSearch &&
    searchQuery.isFetched &&
    !searchQuery.isFetching &&
    searchData !== undefined &&
    searchData.matched_count === 0;

  const renderBody = () => {
    if (defaultModelsQuery.isError) {
      return <Empty description="Could not load models for this key" />;
    }

    if (nothingFound) {
      return <Empty description={`Nothing found for "${searchInput.trim()}"`} />;
    }

    if (searchQuery.isError && hasActiveSearch) {
      return <Empty description="Could not load search results" />;
    }

    if (sections && sections.length > 0) {
      return (
        <>
          {truncated ? (
            <Alert
              type="info"
              showIcon
              className="mb-3"
              message="Results capped — refine your search to narrow matches."
            />
          ) : null}
          {sections.map((section) => (
            <SectionBlock key={`${section.section_kind}-${section.title}`} section={section} warnNoTeam={warnNoTeam} />
          ))}
        </>
      );
    }

    if (!isInitialLoading && defaultModelsQuery.isSuccess && (!sections || sections.length === 0)) {
      return <Empty description="No models resolved for this key" />;
    }

    return null;
  };

  return (
    <Card
      title="Models"
      loading={isInitialLoading}
      styles={{
        body: {
          padding: 0,
          display: 'flex',
          flexDirection: 'column',
          maxHeight: CARD_BODY_MAX_HEIGHT,
          overflow: 'hidden',
        },
      }}
    >
      <div className="shrink-0 border-b border-gray-100 px-4 pb-3 pt-3">
        <Input.Search
          allowClear
          placeholder="Filter models…"
          value={searchInput}
          loading={searchInputLoading}
          onChange={(e) => setSearchInput(e.target.value)}
          onSearch={(v) => setSearchInput(v)}
        />
      </div>
      <div
        data-testid="key-model-list-scroll"
        className="min-h-0 flex-1 overflow-y-auto overflow-x-hidden px-4 pb-4 pt-2"
      >
        {renderBody()}
      </div>
    </Card>
  );
};

export default KeyModelList;
