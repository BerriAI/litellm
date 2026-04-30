import React from 'react';
import Link from '@docusaurus/Link';
import styles from './styles.module.css';

export default function NavigationCards({ items, columns = 2 }) {
  return (
    <div
      className={styles.grid}
      style={{ '--nav-columns': columns }}
    >
      {items.map((item, i) => {
        const isExternal =
          item.to && (item.to.startsWith('http://') || item.to.startsWith('https://'));
        return (
          <Link
            key={i}
            to={item.to}
            className={styles.card}
            target={isExternal ? '_blank' : undefined}
            rel={isExternal ? 'noopener noreferrer' : undefined}
          >
            {item.icon && (
              <div className={styles.icon}>{item.icon}</div>
            )}
            <div className={styles.title}>{item.title}</div>
            {item.description && (
              <div className={styles.description}>{item.description}</div>
            )}
            {item.listDescription && (
              <ul className={styles.list}>
                {item.listDescription.map((line, j) => (
                  <li key={j}>{line}</li>
                ))}
              </ul>
            )}
            {isExternal && (
              <span className={styles.externalIcon}>↗</span>
            )}
          </Link>
        );
      })}
    </div>
  );
}
