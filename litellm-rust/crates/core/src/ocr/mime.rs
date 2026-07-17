use crate::constants::{
    MIME_APPLICATION_OCTET_STREAM, MIME_APPLICATION_PDF, MIME_BINARY_OCTET_STREAM, MIME_IMAGE_BMP,
    MIME_IMAGE_GIF, MIME_IMAGE_JPEG, MIME_IMAGE_PNG, MIME_IMAGE_TIFF, MIME_IMAGE_WEBP,
};

pub fn sniff_document_mime(bytes: &[u8]) -> Option<&'static str> {
    if bytes.starts_with(b"%PDF-") {
        return Some(MIME_APPLICATION_PDF);
    }
    if bytes.starts_with(&[0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a]) {
        return Some(MIME_IMAGE_PNG);
    }
    if bytes.starts_with(&[0xff, 0xd8, 0xff]) {
        return Some(MIME_IMAGE_JPEG);
    }
    if bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a") {
        return Some(MIME_IMAGE_GIF);
    }
    if bytes.len() >= 12 && bytes.starts_with(b"RIFF") && &bytes[8..12] == b"WEBP" {
        return Some(MIME_IMAGE_WEBP);
    }
    if bytes.starts_with(&[0x49, 0x49, 0x2a, 0x00]) || bytes.starts_with(&[0x4d, 0x4d, 0x00, 0x2a])
    {
        return Some(MIME_IMAGE_TIFF);
    }
    None
}

pub fn mime_from_file_name(file_name: &str) -> Option<&'static str> {
    let base_name = file_name.rsplit(['/', '\\']).next()?;
    let extension = base_name.rsplit_once('.')?.1.to_ascii_lowercase();
    match extension.as_str() {
        "pdf" => Some(MIME_APPLICATION_PDF),
        "png" => Some(MIME_IMAGE_PNG),
        "jpg" | "jpeg" => Some(MIME_IMAGE_JPEG),
        "gif" => Some(MIME_IMAGE_GIF),
        "webp" => Some(MIME_IMAGE_WEBP),
        "tiff" | "tif" => Some(MIME_IMAGE_TIFF),
        "bmp" => Some(MIME_IMAGE_BMP),
        _ => None,
    }
}

pub fn normalize_declared_mime(declared: &str) -> Option<String> {
    let base = declared
        .split(';')
        .next()
        .unwrap_or("")
        .trim()
        .to_ascii_lowercase();
    if base.is_empty() || base == MIME_APPLICATION_OCTET_STREAM || base == MIME_BINARY_OCTET_STREAM
    {
        return None;
    }
    Some(base)
}

pub fn resolve_document_mime(
    declared: Option<&str>,
    bytes: &[u8],
    file_name: Option<&str>,
) -> String {
    if let Some(specific) = declared.and_then(normalize_declared_mime) {
        return specific;
    }
    sniff_document_mime(bytes)
        .or_else(|| file_name.and_then(mime_from_file_name))
        .map(str::to_string)
        .unwrap_or_else(|| MIME_APPLICATION_OCTET_STREAM.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sniff_document_mime_detects_supported_signatures() {
        assert_eq!(
            sniff_document_mime(b"%PDF-1.7\ncontent"),
            Some(MIME_APPLICATION_PDF)
        );
        assert_eq!(
            sniff_document_mime(&[0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a, 0x00]),
            Some(MIME_IMAGE_PNG)
        );
        assert_eq!(
            sniff_document_mime(&[0xff, 0xd8, 0xff, 0xe0]),
            Some(MIME_IMAGE_JPEG)
        );
        assert_eq!(sniff_document_mime(b"GIF87a....."), Some(MIME_IMAGE_GIF));
        assert_eq!(sniff_document_mime(b"GIF89a....."), Some(MIME_IMAGE_GIF));
        assert_eq!(
            sniff_document_mime(b"RIFF\x00\x00\x00\x00WEBPVP8 "),
            Some(MIME_IMAGE_WEBP)
        );
        assert_eq!(
            sniff_document_mime(&[0x49, 0x49, 0x2a, 0x00]),
            Some(MIME_IMAGE_TIFF)
        );
        assert_eq!(
            sniff_document_mime(&[0x4d, 0x4d, 0x00, 0x2a]),
            Some(MIME_IMAGE_TIFF)
        );
    }

    #[test]
    fn sniff_document_mime_does_not_detect_bmp_by_magic_bytes() {
        assert_eq!(sniff_document_mime(b"BMxxxx"), None);
        assert_eq!(sniff_document_mime(b"BM\x36\x00\x00\x00random"), None);
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
        assert_eq!(
            mime_from_file_name("report.pdf"),
            Some(MIME_APPLICATION_PDF)
        );
        assert_eq!(mime_from_file_name("/docs/scan.PNG"), Some(MIME_IMAGE_PNG));
        assert_eq!(mime_from_file_name("a/b/photo.jpeg"), Some(MIME_IMAGE_JPEG));
        assert_eq!(mime_from_file_name("photo.JPG"), Some(MIME_IMAGE_JPEG));
        assert_eq!(mime_from_file_name("anim.gif"), Some(MIME_IMAGE_GIF));
        assert_eq!(mime_from_file_name("sticker.webp"), Some(MIME_IMAGE_WEBP));
        assert_eq!(mime_from_file_name("scan.tif"), Some(MIME_IMAGE_TIFF));
        assert_eq!(mime_from_file_name("scan.tiff"), Some(MIME_IMAGE_TIFF));
        assert_eq!(mime_from_file_name("bitmap.bmp"), Some(MIME_IMAGE_BMP));
    }

    #[test]
    fn mime_from_file_name_returns_none_when_unmapped_or_missing_extension() {
        assert_eq!(mime_from_file_name("archive.zip"), None);
        assert_eq!(mime_from_file_name("noextension"), None);
        assert_eq!(mime_from_file_name("/trailing/"), None);
        assert_eq!(mime_from_file_name(""), None);
    }

    #[test]
    fn normalize_declared_mime_lowercases_and_strips_parameters() {
        assert_eq!(
            normalize_declared_mime("Application/PDF"),
            Some(MIME_APPLICATION_PDF.to_string())
        );
        assert_eq!(
            normalize_declared_mime("image/png; charset=utf-8"),
            Some(MIME_IMAGE_PNG.to_string())
        );
    }

    #[test]
    fn normalize_declared_mime_treats_generic_and_blank_as_absent() {
        assert_eq!(normalize_declared_mime("application/octet-stream"), None);
        assert_eq!(normalize_declared_mime("Binary/Octet-Stream"), None);
        assert_eq!(
            normalize_declared_mime("application/octet-stream; charset=binary"),
            None
        );
        assert_eq!(normalize_declared_mime("   "), None);
    }

    #[test]
    fn resolve_document_mime_prefers_specific_declared() {
        assert_eq!(
            resolve_document_mime(Some("image/png"), b"%PDF-1.4", Some("/x.pdf")),
            MIME_IMAGE_PNG
        );
    }

    #[test]
    fn resolve_document_mime_normalizes_declared_casing_and_parameters() {
        assert_eq!(
            resolve_document_mime(Some("Application/PDF; charset=utf-8"), b"", Some("/x")),
            MIME_APPLICATION_PDF
        );
    }

    #[test]
    fn resolve_document_mime_sniffs_when_declared_is_generic() {
        assert_eq!(
            resolve_document_mime(Some("Binary/Octet-Stream"), b"%PDF-1.4 payload", Some("/x")),
            MIME_APPLICATION_PDF
        );
    }

    #[test]
    fn resolve_document_mime_uses_file_name_when_declared_and_bytes_ambiguous() {
        assert_eq!(
            resolve_document_mime(None, b"unrecognized bytes", Some("/docs/file.pdf")),
            MIME_APPLICATION_PDF
        );
    }

    #[test]
    fn resolve_document_mime_falls_back_to_octet_stream() {
        assert_eq!(
            resolve_document_mime(None, b"unrecognized bytes", Some("/docs/file")),
            MIME_APPLICATION_OCTET_STREAM
        );
        assert_eq!(
            resolve_document_mime(
                Some("application/octet-stream"),
                b"unrecognized bytes",
                None
            ),
            MIME_APPLICATION_OCTET_STREAM
        );
    }
}
