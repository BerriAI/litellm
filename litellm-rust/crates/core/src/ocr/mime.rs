use crate::constants::{
    MIME_APPLICATION_PDF, MIME_IMAGE_BMP, MIME_IMAGE_GIF, MIME_IMAGE_JPEG, MIME_IMAGE_PNG,
    MIME_IMAGE_TIFF, MIME_IMAGE_WEBP,
};

pub fn sniff_mime(bytes: &[u8]) -> Option<&'static str> {
    if bytes.starts_with(b"%PDF-") {
        return Some(MIME_APPLICATION_PDF);
    }
    if bytes.starts_with(&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]) {
        return Some(MIME_IMAGE_PNG);
    }
    if bytes.starts_with(&[0xFF, 0xD8, 0xFF]) {
        return Some(MIME_IMAGE_JPEG);
    }
    if bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a") {
        return Some(MIME_IMAGE_GIF);
    }
    if bytes.len() >= 12 && bytes.starts_with(b"RIFF") && &bytes[8..12] == b"WEBP" {
        return Some(MIME_IMAGE_WEBP);
    }
    if bytes.starts_with(&[0x49, 0x49, 0x2A, 0x00]) || bytes.starts_with(&[0x4D, 0x4D, 0x00, 0x2A])
    {
        return Some(MIME_IMAGE_TIFF);
    }
    if bytes.starts_with(b"BM") {
        return Some(MIME_IMAGE_BMP);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sniffs_pdf() {
        assert_eq!(sniff_mime(b"%PDF-1.7\n..."), Some(MIME_APPLICATION_PDF));
    }

    #[test]
    fn does_not_sniff_pdf_without_version_marker() {
        assert_eq!(sniff_mime(b"%PDFxx"), None);
    }

    #[test]
    fn sniffs_png() {
        assert_eq!(
            sniff_mime(&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A, 0x00]),
            Some(MIME_IMAGE_PNG)
        );
    }

    #[test]
    fn sniffs_jpeg() {
        assert_eq!(sniff_mime(&[0xFF, 0xD8, 0xFF, 0xE0]), Some(MIME_IMAGE_JPEG));
    }

    #[test]
    fn sniffs_gif() {
        assert_eq!(sniff_mime(b"GIF87a...."), Some(MIME_IMAGE_GIF));
        assert_eq!(sniff_mime(b"GIF89a...."), Some(MIME_IMAGE_GIF));
    }

    #[test]
    fn sniffs_webp() {
        assert_eq!(
            sniff_mime(b"RIFF\x00\x00\x00\x00WEBP"),
            Some(MIME_IMAGE_WEBP)
        );
    }

    #[test]
    fn does_not_sniff_riff_without_webp() {
        assert_eq!(sniff_mime(b"RIFF\x00\x00\x00\x00WAVE"), None);
    }

    #[test]
    fn sniffs_tiff_both_byte_orders() {
        assert_eq!(sniff_mime(&[0x49, 0x49, 0x2A, 0x00]), Some(MIME_IMAGE_TIFF));
        assert_eq!(sniff_mime(&[0x4D, 0x4D, 0x00, 0x2A]), Some(MIME_IMAGE_TIFF));
    }

    #[test]
    fn sniffs_bmp() {
        assert_eq!(sniff_mime(b"BM...."), Some(MIME_IMAGE_BMP));
    }

    #[test]
    fn returns_none_for_unknown_and_truncated() {
        assert_eq!(sniff_mime(&[0x00, 0x01, 0x02, 0x03]), None);
        assert_eq!(sniff_mime(b""), None);
        assert_eq!(sniff_mime(&[0x89, b'P', b'N', b'G']), None);
    }
}
