use std::{error::Error, fmt, ops::Deref, sync::Arc};

pub struct Report<'a>(&'a (dyn Error + 'static));

impl<'a> Report<'a> {
    pub fn new(error: &'a (dyn Error + 'static)) -> Self {
        Self(error)
    }
}

impl fmt::Display for Report<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}", self.0)?;
        std::iter::successors(self.0.source(), |&cause| cause.source())
            .try_for_each(|cause| write!(formatter, ": {cause}"))
    }
}

pub fn find<'a, T: Error + 'static>(error: &'a (dyn Error + 'static)) -> Option<&'a T> {
    std::iter::successors(Some(error), |&cause| cause.source()).find_map(|cause| {
        cause
            .downcast_ref::<T>()
            .or_else(|| cause.downcast_ref::<Source>()?.downcast_ref::<T>())
    })
}

#[derive(Clone, Debug, thiserror::Error)]
#[error(transparent)]
pub struct Source(Arc<dyn Error + Send + Sync>);

impl Source {
    pub fn new(error: impl Error + Send + Sync + 'static) -> Self {
        Self(Arc::new(error))
    }
}

impl From<Box<dyn Error + Send + Sync>> for Source {
    fn from(error: Box<dyn Error + Send + Sync>) -> Self {
        Self(Arc::from(error))
    }
}

impl Deref for Source {
    type Target = dyn Error + Send + Sync;

    fn deref(&self) -> &Self::Target {
        self.0.as_ref()
    }
}

impl PartialEq for Source {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for Source {}

#[cfg(test)]
mod tests {
    use std::io;

    use rstest::rstest;

    use super::{Report, Source, find};

    #[derive(Debug, thiserror::Error)]
    enum Leaf {
        #[error("could not read the config")]
        Read(#[source] io::Error),
        #[error("could not parse the config")]
        Parse(#[source] Source),
        #[error("config is empty")]
        Empty,
    }

    #[derive(Debug, thiserror::Error)]
    enum Outer {
        #[error("startup failed")]
        Startup(#[from] Leaf),
        #[error(transparent)]
        Nested(Leaf),
    }

    fn denied() -> io::Error {
        io::Error::new(io::ErrorKind::PermissionDenied, "permission denied")
    }

    #[rstest]
    #[case::leaf(Outer::Nested(Leaf::Empty), "config is empty")]
    #[case::context_over_a_cause(
        Outer::Startup(Leaf::Read(denied())),
        "startup failed: could not read the config: permission denied"
    )]
    #[case::transparent_adds_no_text(
        Outer::Nested(Leaf::Read(denied())),
        "could not read the config: permission denied"
    )]
    #[case::erased_cause_keeps_its_chain(
        Outer::Nested(Leaf::Parse(Source::new(Leaf::Read(denied())))),
        "could not parse the config: could not read the config: permission denied"
    )]
    fn report_renders_every_layer_once(#[case] error: Outer, #[case] expected: &str) {
        assert_eq!(Report::new(&error).to_string(), expected);
    }

    #[rstest]
    #[case::direct_source(Outer::Startup(Leaf::Read(denied())))]
    #[case::behind_an_erased_source(Outer::Startup(Leaf::Parse(Source::new(Leaf::Read(
        denied()
    )))))]
    #[case::behind_a_boxed_source(Outer::Startup(Leaf::Parse(Source::from(
        Box::<dyn std::error::Error + Send + Sync>::from(denied())
    ))))]
    #[case::behind_a_transparent_layer(Outer::Nested(Leaf::Parse(Source::new(denied()))))]
    fn find_reaches_the_typed_cause_anywhere_in_the_chain(#[case] error: Outer) {
        let cause = find::<io::Error>(&error).expect("io cause");
        assert_eq!(cause.kind(), io::ErrorKind::PermissionDenied);
    }

    #[test]
    fn find_answers_none_when_the_chain_has_no_such_cause() {
        assert!(find::<io::Error>(&Outer::Nested(Leaf::Empty)).is_none());
    }

    #[test]
    fn a_clone_shares_its_cause() {
        let source = Source::new(denied());
        assert_eq!(source.clone(), source);
        assert_ne!(Source::new(denied()), source);
    }
}
