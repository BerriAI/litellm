import React, {useEffect, useMemo, useRef, useState} from 'react';
import OriginalDocItem from '@theme-original/DocItem';
import styles from './styles.module.css';

function normalizeSourcePath(source) {
  if (!source || typeof source !== 'string') {
    return null;
  }
  if (source.startsWith('@site/')) {
    return source.slice('@site/'.length);
  }
  return null;
}

async function copyTextToClipboard(text) {
  if (
    typeof navigator !== 'undefined' &&
    navigator.clipboard &&
    typeof navigator.clipboard.writeText === 'function'
  ) {
    await navigator.clipboard.writeText(text);
    return;
  }

  if (typeof document === 'undefined') {
    throw new Error('Clipboard API unavailable');
  }

  const textarea = document.createElement('textarea');
  textarea.value = text;
  textarea.setAttribute('readonly', '');
  textarea.style.position = 'absolute';
  textarea.style.left = '-9999px';
  document.body.appendChild(textarea);
  textarea.select();

  const success = document.execCommand('copy');
  document.body.removeChild(textarea);

  if (!success) {
    throw new Error('Fallback copy command failed');
  }
}

function CopyMarkdownButton({metadata}) {
  const timeoutRef = useRef(null);
  const [status, setStatus] = useState('idle');

  const rawMarkdownUrl = useMemo(() => {
    const sourcePath = normalizeSourcePath(metadata?.source);
    if (!sourcePath) {
      return null;
    }

    return `/__source_markdown/${sourcePath}`;
  }, [metadata?.source]);

  const hasSource = Boolean(rawMarkdownUrl);

  const buttonLabel = (() => {
    if (status === 'loading') {
      return 'Copying...';
    }
    if (status === 'copied') {
      return 'Copied markdown';
    }
    if (status === 'error') {
      return 'Retry copy markdown';
    }
    return 'Copy markdown';
  })();

  const onCopyMarkdown = async () => {
    if (!rawMarkdownUrl || status === 'loading') {
      return;
    }

    setStatus('loading');
    try {
      const response = await fetch(rawMarkdownUrl, {
        headers: {'Accept': 'text/plain'},
      });
      if (!response.ok) {
        throw new Error(`Failed to load markdown: ${response.status}`);
      }
      const markdown = await response.text();
      await copyTextToClipboard(markdown);
      setStatus('copied');

      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
      timeoutRef.current = setTimeout(() => {
        setStatus('idle');
      }, 2000);
    } catch (_error) {
      setStatus('error');
    }
  };

  useEffect(() => {
    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, []);

  return (
    <div className={styles.copyMarkdownToolbar}>
      <button
        type="button"
        className={styles.copyMarkdownButton}
        onClick={onCopyMarkdown}
        disabled={!hasSource || status === 'loading'}
        title={
          hasSource
            ? 'Copy this page markdown to your clipboard'
            : 'Markdown source path unavailable for this page'
        }>
        {buttonLabel}
      </button>
    </div>
  );
}

export default function DocItem(props) {
  const metadata = props?.content?.metadata;

  return (
    <>
      <CopyMarkdownButton metadata={metadata} />
      <OriginalDocItem {...props} />
    </>
  );
}
