import React, {useEffect} from 'react';
import OriginalBlogPostPage from '@theme-original/BlogPostPage';
import styles from './styles.module.css';

function BackLink() {
  return (
    <div className={styles.backOuter}>
      <a href="/blog" className={styles.backLink}>
        <svg className={styles.backArrow} fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16l-4-4m0 0l4-4m-4 4h18" />
        </svg>
        Blog
      </a>
    </div>
  );
}

function HiringCTA() {
  return (
    <div className={styles.ctaOuter}>
      <div className={styles.cta}>
        <p className={styles.ctaEyebrow}>We're hiring</p>
        <a
          href="https://jobs.ashbyhq.com/litellm"
          target="_blank"
          rel="noopener noreferrer"
          className={styles.ctaLink}
        >
          Like what you see? Join us
          <svg className={styles.ctaArrow} fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 8l4 4m0 0l-4 4m4-4H3" />
          </svg>
        </a>
        <p className={styles.ctaSub}>Come build the future of AI infrastructure.</p>
      </div>
    </div>
  );
}

export default function BlogPostPage(props) {
  // Add body class so CSS can hide the sidebar
  useEffect(() => {
    document.body.classList.add('blog-post-body');
    return () => document.body.classList.remove('blog-post-body');
  }, []);

  return (
    <>
      <BackLink />
      <OriginalBlogPostPage {...props} />
      <HiringCTA />
    </>
  );
}
