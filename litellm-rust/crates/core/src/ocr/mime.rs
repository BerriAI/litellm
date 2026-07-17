pub fn sniff_document_mime(bytes: &[u8]) -> Option<&'static str> {
    if bytes.starts_with(b"%PDF-") {
        return Some("application/pdf");
    }
    if bytes.starts_with(&[0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a]) {
        return Some("image/png");
    }
    if bytes.starts_with(&[0xff, 0xd8, 0xff]) {
        return Some("image/jpeg");
    }
    if bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a") {
        return Some("image/gif");
    }
    if bytes.len() >= 12 && bytes.starts_with(b"RIFF") && &bytes[8..12] == b"WEBP" {
        return Some("image/webp");
    }
    if bytes.starts_with(&[0x49, 0x49, 0x2a, 0x00]) || bytes.starts_with(&[0x4d, 0x4d, 0x00, 0x2a])
    {
        return Some("image/tiff");
    }
    if bytes.starts_with(b"BM") {
        return Some("image/bmp");
    }
    None
}

pub fn mime_from_file_name(file_name: &str) -> Option<&'static str> {
    let base_name = file_name.rsplit(['/', '\\']).next()?;
    let extension = base_name.rsplit_once('.')?.1.to_ascii_lowercase();
    match extension.as_str() {
        "pdf" => Some("application/pdf"),
        "png" => Some("image/png"),
        "jpg" | "jpeg" => Some("image/jpeg"),
        "gif" => Some("image/gif"),
        "webp" => Some("image/webp"),
        "tiff" | "tif" => Some("image/tiff"),
        "bmp" => Some("image/bmp"),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sniff_document_mime_detects_supported_signatures() {
        assert_eq!(
            sniff_document_mime(b"%PDF-1.7\ncontent"),
            Some("application/pdf")
        );
        assert_eq!(
            sniff_document_mime(&[0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a, 0x00]),
            Some("image/png")
        );
        assert_eq!(
            sniff_document_mime(&[0xff, 0xd8, 0xff, 0xe0]),
            Some("image/jpeg")
        );
        assert_eq!(sniff_document_mime(b"GIF87a....."), Some("image/gif"));
        assert_eq!(sniff_document_mime(b"GIF89a....."), Some("image/gif"));
        assert_eq!(
            sniff_document_mime(b"RIFF\x00\x00\x00\x00WEBPVP8 "),
            Some("image/webp")
        );
        assert_eq!(
            sniff_document_mime(&[0x49, 0x49, 0x2a, 0x00]),
            Some("image/tiff")
        );
        assert_eq!(
            sniff_document_mime(&[0x4d, 0x4d, 0x00, 0x2a]),
            Some("image/tiff")
        );
        assert_eq!(sniff_document_mime(b"BMxxxx"), Some("image/bmp"));
    }

    #[test]
    fn sniff_document_mime_returns_none_for_unknown_or_riff_non_webp() {
        assert_eq!(sniff_document_mime(b"plain text payload"), None);
        assert_eq!(sniff_document_mime(b"RIFF\x00\x00\x00\x00WAVEfmt "), None);
        assert_eq!(sniff_document_mime(b""), None);
        assert_eq!(sniff_document_mime(b"RIFF"), None);
    }

    #[test]
    fn mime_from_file_name_maps_known_suffixes() {
        assert_eq!(mime_from_file_name("report.pdf"), Some("application/pdf"));
        assert_eq!(mime_from_file_name("/docs/scan.PNG"), Some("image/png"));
        assert_eq!(mime_from_file_name("a/b/photo.jpeg"), Some("image/jpeg"));
        assert_eq!(mime_from_file_name("photo.JPG"), Some("image/jpeg"));
        assert_eq!(mime_from_file_name("anim.gif"), Some("image/gif"));
        assert_eq!(mime_from_file_name("sticker.webp"), Some("image/webp"));
        assert_eq!(mime_from_file_name("scan.tif"), Some("image/tiff"));
        assert_eq!(mime_from_file_name("scan.tiff"), Some("image/tiff"));
        assert_eq!(mime_from_file_name("bitmap.bmp"), Some("image/bmp"));
    }

    #[test]
    fn mime_from_file_name_returns_none_when_unmapped_or_missing_extension() {
        assert_eq!(mime_from_file_name("archive.zip"), None);
        assert_eq!(mime_from_file_name("noextension"), None);
        assert_eq!(mime_from_file_name("/trailing/"), None);
        assert_eq!(mime_from_file_name(""), None);
    }
}
