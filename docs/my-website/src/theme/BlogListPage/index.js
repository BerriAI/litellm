import React from 'react';
import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';
import styles from './styles.module.css';

// ── Provider marquee ──────────────────────────────────────────────────────
const PROVIDERS = [
  { name: 'OpenAI',        img: 'https://www.google.com/s2/favicons?domain=openai.com&sz=64' },
  { name: 'Anthropic',     img: 'https://www.google.com/s2/favicons?domain=claude.ai&sz=64' },
  { name: 'Google Gemini', img: 'https://www.google.com/s2/favicons?domain=ai.google.dev&sz=64' },
  { name: 'AWS Bedrock',   img: 'https://www.google.com/s2/favicons?domain=aws.amazon.com&sz=64' },
  { name: 'Azure OpenAI',  img: 'https://www.google.com/s2/favicons?domain=azure.microsoft.com&sz=64' },
  { name: 'Mistral AI',    img: 'https://www.google.com/s2/favicons?domain=mistral.ai&sz=64' },
  { name: 'Meta Llama',    img: 'https://www.google.com/s2/favicons?domain=meta.com&sz=64' },
  { name: 'Groq',          img: 'https://www.google.com/s2/favicons?domain=groq.com&sz=64' },
  { name: 'Hugging Face',  img: 'https://www.google.com/s2/favicons?domain=huggingface.co&sz=64' },
  { name: 'Perplexity',    img: 'https://www.google.com/s2/favicons?domain=perplexity.ai&sz=64' },
  { name: 'DeepSeek',      img: 'https://www.google.com/s2/favicons?domain=deepseek.com&sz=64' },
  { name: 'Cohere',        img: 'https://www.google.com/s2/favicons?domain=cohere.com&sz=64' },
  { name: 'Together AI',   img: 'https://www.google.com/s2/favicons?domain=together.ai&sz=64' },
  { name: 'Vertex AI',     img: 'https://www.google.com/s2/favicons?domain=cloud.google.com&sz=64' },
];

const DOUBLED = [...PROVIDERS, ...PROVIDERS];

function ProviderMarquee() {
  return (
    <div className={styles.marqueeWrap}>
      <p className={styles.marqueeLabel}>Routing to 100+ providers</p>
      <div className={styles.marqueeOuter}>
        <div className={styles.fadeLeft} />
        <div className={styles.fadeRight} />
        <div className={styles.marqueeTrack}>
          {DOUBLED.map((p, i) => (
            <span key={i} className={styles.marqueeItem}>
              <img src={p.img} alt={p.name} width={18} height={18} className={styles.marqueeIcon} />
              <span>{p.name}</span>
              <span className={styles.marqueeSep}>|</span>
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}

// ── Post row ──────────────────────────────────────────────────────────────
function formatDate(dateStr) {
  return new Date(dateStr).toLocaleDateString('en-US', {
    month: 'long', day: 'numeric', year: 'numeric',
  });
}

function AuthorList({authors}) {
  if (!authors || authors.length === 0) return null;
  return (
    <>
      {authors.map((a, i) => (
        <React.Fragment key={a.name}>
          {i > 0 && <span className={styles.authorSep}> </span>}
          {a.url ? (
            <a href={a.url} target="_blank" rel="noopener" className={styles.authorLink}>{a.name}</a>
          ) : (
            <span className={styles.authorName}>{a.name}</span>
          )}
        </React.Fragment>
      ))}
    </>
  );
}

function PostRow({post}) {
  const {title, permalink, date, description, authors} = post;
  return (
    <article className={styles.post}>
      <Link to={permalink} className={styles.titleLink}>
        <h2 className={styles.title}>{title}</h2>
      </Link>
      {description && <p className={styles.desc}>{description}</p>}
      <div className={styles.meta}>
        <AuthorList authors={authors} />
        {authors && authors.length > 0 && <span className={styles.metaDash}> — </span>}
        <time className={styles.date} dateTime={date}>{formatDate(date)}</time>
      </div>
    </article>
  );
}

function Pagination({metadata}) {
  const {previousPage, nextPage} = metadata;
  if (!previousPage && !nextPage) return null;
  return (
    <nav className={styles.pagination} aria-label="Blog list pagination">
      {previousPage ? <Link to={previousPage} className={styles.pageLink}>&larr; Newer posts</Link> : <span />}
      {nextPage ? <Link to={nextPage} className={styles.pageLink}>Older posts &rarr;</Link> : <span />}
    </nav>
  );
}

// ── Page ──────────────────────────────────────────────────────────────────
export default function BlogListPage(props) {
  const items = props.items || [];
  const metadata = props.metadata || {};

  return (
    <Layout
      title="Engineering Blog"
      description="How we build the world's most widely used open-source AI Gateway. Routing, reliability, observability, and what we learn along the way."
    >
      <div className={styles.page}>
        {/* Hero */}
        <header className={styles.hero}>
          <p className={styles.eyebrow}>AI Gateway</p>
          <h1 className={styles.heroTitle}>Engineering</h1>
          <p className={styles.heroSub}>
            How we build the world's most widely used open-source AI Gateway.
            Routing, reliability, observability, and what we learn along the way.
          </p>
          <a href="https://jobs.ashbyhq.com/litellm" target="_blank" rel="noopener noreferrer" className={styles.hiringBtn}>
            We're hiring!
          </a>
        </header>

        <ProviderMarquee />

        {/* Post list */}
        <main className={styles.list}>
          {items.map(({content}) => (
            <PostRow key={content.metadata.permalink} post={content.metadata} />
          ))}
        </main>

        <Pagination metadata={metadata} />
      </div>
    </Layout>
  );
}
