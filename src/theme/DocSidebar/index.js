import React from 'react';
import DocSidebarDesktop from '@theme/DocSidebar/Desktop';
import DocSidebarMobile from '@theme/DocSidebar/Mobile';
import SearchBar from '@theme/SearchBar';
import styles from './styles.module.css';

export default function DocSidebar(props) {
  return (
    <>
      <div className={styles.sidebarDesktop}>
        <div className={styles.sidebarContainer}>
          <div className={styles.searchBarSection}>
            <div className={styles.searchBarInner}>
              <SearchBar />
            </div>
          </div>
          <div className={styles.sidebarScroll}>
            <DocSidebarDesktop {...props} />
          </div>
        </div>
      </div>
      <div className={styles.sidebarMobile}>
        <DocSidebarMobile {...props} />
      </div>
    </>
  );
}
