const sharp = require('sharp');
const fs = require('fs');
const path = require('path');

const QUALITY = 75;
const EXTENSIONS = new Set(['.png', '.jpg', '.jpeg']);

function walk(dir) {
  if (!fs.existsSync(dir)) return [];
  const files = [];
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) files.push(...walk(full));
    else if (EXTENSIONS.has(path.extname(entry.name).toLowerCase())) files.push(full);
  }
  return files;
}

async function optimizeFile(filePath) {
  const ext = path.extname(filePath).toLowerCase();
  const tmp = filePath + '.opt';
  try {
    const pipeline = sharp(filePath);
    if (ext === '.png') {
      await pipeline.png({ quality: QUALITY, compressionLevel: 9 }).toFile(tmp);
    } else {
      await pipeline.jpeg({ quality: QUALITY, mozjpeg: true }).toFile(tmp);
    }
    const orig = fs.statSync(filePath).size;
    const next = fs.statSync(tmp).size;
    if (next < orig) {
      fs.renameSync(tmp, filePath);
      return orig - next;
    }
    fs.unlinkSync(tmp);
    return 0;
  } catch {
    if (fs.existsSync(tmp)) fs.unlinkSync(tmp);
    return 0;
  }
}

module.exports = function optimizeImagesPlugin() {
  return {
    name: 'optimize-images',
    async postBuild({ outDir }) {
      const files = walk(outDir);
      if (!files.length) return;
      let saved = 0;
      await Promise.all(files.map(async (f) => { saved += await optimizeFile(f); }));
      const mb = (saved / 1024 / 1024).toFixed(1);
      console.log(`\n[optimize-images] Compressed ${files.length} images, saved ${mb} MB`);
    },
  };
};
