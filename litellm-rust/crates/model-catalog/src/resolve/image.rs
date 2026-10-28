/// What an image call asks for. `resolve_image` builds main's eight key spellings from it
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImageCandidates {
    pub model: String,
    pub provider: Option<String>,
    /// Width and height, `1024 x 1024` when the request names none, as on main
    pub size: Option<(u32, u32)>,
    pub quality: Option<String>,
}

/// A size and quality keyed row, `{quality}/{W}-x-{H}/{model}`, parsed once per index build for fal's nearest match
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImageKey {
    pub quality: Option<String>,
    pub width: u32,
    pub height: u32,
    pub model: String,
}

impl ImageCandidates {
    /// main's `default_image_cost_calculator` key spellings, in its order
    pub fn spellings(&self) -> Vec<String> {
        let (first, second) = self.size.unwrap_or((1024, 1024));
        let size = format!("{first}-x-{second}");
        let provider_prefix = self
            .provider
            .as_deref()
            .map(|provider| format!("{provider}/"));
        let without_provider = provider_prefix
            .as_deref()
            .and_then(|prefix| self.model.strip_prefix(prefix))
            .map(|rest| rest.replace(provider_prefix.as_deref().unwrap_or_default(), ""));
        let base = match (&self.provider, &without_provider) {
            (Some(provider), Some(rest)) => format!("{provider}/{size}/{rest}"),
            _ => format!("{size}/{}", self.model),
        };
        let quality = self
            .quality
            .as_deref()
            .filter(|quality| !quality.is_empty());
        let with_quality = |name: &str| {
            quality.map_or_else(|| name.to_owned(), |quality| format!("{quality}/{name}"))
        };
        let provider_first = quality
            .zip(self.provider.as_deref())
            .map(|(quality, provider)| {
                format!(
                    "{provider}/{quality}/{size}/{}",
                    without_provider.as_deref().unwrap_or(&self.model)
                )
            });
        let last_segment = self.model.rsplit('/').next().unwrap_or(&self.model);
        let model_without_provider = format!("{size}/{last_segment}");
        [
            Some(with_quality(&base)),
            provider_first,
            Some(base.clone()),
            Some(format!("high/{base}")),
            Some(with_quality(&model_without_provider)),
            Some(model_without_provider.clone()),
            Some(self.model.clone()),
            without_provider,
        ]
        .into_iter()
        .flatten()
        .collect()
    }
}

impl ImageKey {
    /// A `{provider}/{quality}/{W}-x-{H}/{model}` key, `None` for any other shape
    pub fn parse(key: &str, provider: &str) -> Option<Self> {
        let rest = key.strip_prefix(provider)?.strip_prefix('/')?;
        let (quality, rest) = rest.split_once('/')?;
        let (size, model) = rest.split_once('/')?;
        let (width, height) = size.split_once("-x-")?;
        let width: u32 = width.parse().ok().filter(|width| *width > 0)?;
        let height: u32 = height.parse().ok().filter(|height| *height > 0)?;
        Some(Self {
            quality: Some(quality.to_owned()),
            width,
            height,
            model: model.to_owned(),
        })
    }

    fn pixels(&self) -> u64 {
        u64::from(self.width) * u64::from(self.height)
    }
}

/// fal's nearest size match. Of the `{provider}/{quality}/{W}-x-{H}/{model}` keys the row filter keeps, the one whose
/// pixel count is closest to the target's, the smaller one on a tie
pub fn nearest_image_key<'a>(
    keys: impl IntoIterator<Item = &'a str>,
    provider: &str,
    model: &str,
    quality: &str,
    target: (u32, u32),
) -> Option<&'a str> {
    let target_pixels = u64::from(target.0) * u64::from(target.1);
    keys.into_iter()
        .filter_map(|key| Some((key, ImageKey::parse(key, provider)?)))
        .filter(|(_, parsed)| parsed.model == model && parsed.quality.as_deref() == Some(quality))
        .min_by_key(|(_, parsed)| (parsed.pixels().abs_diff(target_pixels), parsed.pixels()))
        .map(|(key, _)| key)
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    fn the_spellings_follow_mains_order() {
        let candidates = ImageCandidates {
            model: "azure/dall-e-3".into(),
            provider: Some("azure".into()),
            size: Some((1024, 1792)),
            quality: Some("hd".into()),
        };
        assert_eq!(
            candidates.spellings(),
            [
                "hd/azure/1024-x-1792/dall-e-3",
                "azure/hd/1024-x-1792/dall-e-3",
                "azure/1024-x-1792/dall-e-3",
                "high/azure/1024-x-1792/dall-e-3",
                "hd/1024-x-1792/dall-e-3",
                "1024-x-1792/dall-e-3",
                "azure/dall-e-3",
                "dall-e-3",
            ]
        );
    }

    #[rstest]
    fn without_a_provider_prefix_or_quality_the_spellings_are_the_bare_ones() {
        let candidates = ImageCandidates {
            model: "dall-e-2".into(),
            provider: Some("openai".into()),
            size: None,
            quality: None,
        };
        assert_eq!(
            candidates.spellings(),
            [
                "1024-x-1024/dall-e-2",
                "1024-x-1024/dall-e-2",
                "high/1024-x-1024/dall-e-2",
                "1024-x-1024/dall-e-2",
                "1024-x-1024/dall-e-2",
                "dall-e-2",
            ]
        );
    }

    #[rstest]
    #[case::exact((1024, 1024), "fal_ai/high/1024-x-1024/flux")]
    #[case::nearest((1100, 1100), "fal_ai/high/1024-x-1024/flux")]
    #[case::tie_takes_the_smaller((768, 768), "fal_ai/high/512-x-512/flux")]
    fn the_nearest_size_wins(#[case] target: (u32, u32), #[case] expected: &str) {
        let keys = [
            "fal_ai/high/512-x-512/flux",
            "fal_ai/high/1024-x-1024/flux",
            "fal_ai/low/1024-x-1024/flux",
            "fal_ai/high/1024-x-1024/other",
            "fal_ai/flux",
        ];
        assert_eq!(
            nearest_image_key(keys, "fal_ai", "flux", "high", target),
            Some(expected)
        );
    }
}
