use litellm_tracing::{Logger, Metadata, Record, Sink};
use rstest::rstest;
use std::sync::mpsc;

struct Capture(mpsc::Sender<String>);
impl Sink for Capture {
    fn enabled(&self, _: &Metadata<'_>) -> bool {
        true
    }
    fn emit(&self, record: &Record) {
        self.0.send(record.message.clone()).unwrap();
    }
}

fn emit() {
    let span = tracing::info_span!("scoped registration");
    span.in_scope(|| tracing::info!("scoped event"));
}

#[rstest]
fn callsites_first_seen_without_a_subscriber_remain_visible_to_a_scoped_logger() {
    let (sender, records) = mpsc::channel();
    let logger = Logger::new(Capture(sender));
    std::thread::spawn(emit).join().unwrap();
    assert!(records.try_recv().is_err());
    logger.scope(emit);
    assert_eq!(
        records.try_iter().collect::<Vec<_>>(),
        ["scoped event", "span closed"]
    );
}
