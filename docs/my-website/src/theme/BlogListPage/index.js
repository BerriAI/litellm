import React from 'react';
import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';
import styles from './styles.module.css';

const TAG_COLORS = {
  gemini: {bg: '#d2e3fc', text: '#174ea6', darkBg: '#1a3a5c', darkText: '#8ab4f8'},
  anthropic: {bg: '#fde0c4', text: '#b33d00', darkBg: '#4a2800', darkText: '#ffb74d'},
  claude: {bg: '#fde0c4', text: '#b33d00', darkBg: '#4a2800', darkText: '#ffb74d'},
  llms: {bg: '#c8e6c9', text: '#1b5e20', darkBg: '#1b3d1f', darkText: '#81c784'},
};

function hashHue(str) {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    hash = str.charCodeAt(i) + ((hash << 5) - hash);
  }
  return Math.abs(hash) % 360;
}

function getTagColor(label) {
  const key = label.toLowerCase();
  for (const [k, v] of Object.entries(TAG_COLORS)) {
    if (key === k) return v;
  }
  const hue = hashHue(key);
  return {
    bg: `hsl(${hue}, 40%, 90%)`,
    text: `hsl(${hue}, 60%, 25%)`,
    darkBg: `hsl(${hue}, 40%, 20%)`,
    darkText: `hsl(${hue}, 50%, 75%)`,
  };
}

function formatDate(dateStr) {
  const d = new Date(dateStr);
  const now = new Date();
  const diffDays = Math.floor((now - d) / (1000 * 60 * 60 * 24));
  if (diffDays <= 0) return 'Today';
  if (diffDays === 1) return '1d ago';
  if (diffDays < 30) return `${diffDays}d ago`;
  return d.toLocaleDateString('en-US', {month: 'short', day: 'numeric', year: 'numeric'});
}

function BlogCard({post, featured}) {
  const {title, permalink, date, description, tags} = post;
  const visibleTags = (tags || []).slice(0, 3);

  return (
    <Link to={permalink} className={styles.cardLink} aria-label={title}>
      <article className={featured ? styles.cardFeatured : styles.card}>
        <div className={styles.meta}>
          <time className={styles.time} dateTime={date}>{formatDate(date)}</time>
          {featured && <span className={styles.badge}>Latest</span>}
        </div>
        <h2 className={styles.title}>{title}</h2>
        {description && <p className={styles.desc}>{description}</p>}
        {visibleTags.length > 0 && (
          <div className={styles.tags}>
            {visibleTags.map(tag => {
              const c = getTagColor(tag.label);
              return (
                <span key={tag.label} className={styles.tag} style={{
                  '--tag-bg': c.bg, '--tag-text': c.text,
                  '--tag-bg-dark': c.darkBg, '--tag-text-dark': c.darkText,
                }}>{tag.label}</span>
              );
            })}
          </div>
        )}
        <div className={styles.arrow} aria-hidden="true">
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
            <path d="M6 3l5 5-5 5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
        </div>
      </article>
    </Link>
  );
}

function Pagination({metadata}) {
  const {previousPage, nextPage} = metadata;
  if (!previousPage && !nextPage) return null;
  return (
    <nav className={styles.pagination} aria-label="Blog list pagination">
      {previousPage ? (
        <Link to={previousPage} className={styles.paginationLink}>&larr; Newer posts</Link>
      ) : <span />}
      {nextPage ? (
        <Link to={nextPage} className={styles.paginationLink}>Older posts &rarr;</Link>
      ) : <span />}
    </nav>
  );
}

export default function BlogListPage(props) {
  const items = props.items || [];
  const metadata = props.metadata || {};
  const [first, ...rest] = items;

  return (
    <Layout
      title={metadata.blogTitle || 'Blog'}
      description={metadata.blogDescription || 'Guides, announcements, and best practices from the LiteLLM team.'}
    >
      <header className={styles.hero}>
        <h1 className={styles.heroTitle}>The LiteLLM Blog</h1>
        <p className={styles.heroSubtitle}>Guides, announcements, and best practices from the LiteLLM team.</p>
      </header>

      <main className={styles.grid}>
        {first && (
          <BlogCard post={first.content.metadata} featured />
        )}
        {rest.map(({content}) => (
          <BlogCard key={content.metadata.permalink} post={content.metadata} />
        ))}
      </main>

      <Pagination metadata={metadata} />
    </Layout>
  );
}
