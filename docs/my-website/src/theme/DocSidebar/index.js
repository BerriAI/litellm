import React from 'react';
import { useWindowSize } from '@docusaurus/theme-common';
import DocSidebarDesktop from '@theme/DocSidebar/Desktop';
import DocSidebarMobile from '@theme/DocSidebar/Mobile';
import SearchBar from '@theme/SearchBar';
import styles from './styles.module.css';

export default function DocSidebar(props) {
  const windowSize = useWindowSize();

  const shouldRenderSidebarDesktop =
    windowSize === 'desktop' || windowSize === 'ssr';

  const shouldRenderSidebarMobile = windowSize === 'mobile';

  return (
    <>
      {shouldRenderSidebarDesktop && (
        <div className={styles.sidebarContainer}>
          <div className={styles.searchBarContainer}>
            <SearchBar />
          </div>
          <DocSidebarDesktop {...props} />
        </div>
      )}
      {shouldRenderSidebarMobile && <DocSidebarMobile {...props} />}
    </>
  );
}
