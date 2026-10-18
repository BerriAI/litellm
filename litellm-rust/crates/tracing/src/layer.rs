use std::{sync::Arc, time::Instant};

use serde_json::Map;
use tracing::{
    Dispatch, Event, Metadata, Subscriber,
    span::{Attributes, Id, Record as SpanRecord},
};
use tracing_subscriber::{
    Layer, Registry, filter::dynamic_filter_fn, layer::Context, prelude::*, registry::LookupSpan,
};

use crate::{Emitting, Record, Sink};

pub fn sink_layer<S, R>(sink: S) -> impl Layer<R>
where
    S: Sink,
    R: Subscriber + for<'a> LookupSpan<'a>,
{
    let sink = Arc::new(sink);
    let filter_sink = sink.clone();
    Output(sink).with_filter(dynamic_filter_fn(move |metadata, _| {
        enabled(filter_sink.as_ref(), metadata)
    }))
}

pub(crate) fn dispatch(sink: impl Sink) -> Dispatch {
    let sink = Arc::new(sink);
    let filter_sink = sink.clone();
    Dispatch::new(
        Registry::default()
            .with(Output(sink))
            .with(dynamic_filter_fn(move |metadata, _| {
                enabled(filter_sink.as_ref(), metadata)
            })),
    )
}

fn enabled(sink: &impl Sink, metadata: &Metadata<'_>) -> bool {
    let Some(_guard) = Emitting::enter() else {
        return false;
    };
    sink.enabled(metadata)
}

struct Output<S>(Arc<S>);

struct SpanData {
    record: Record,
    started: Instant,
}

impl<S, R> Layer<R> for Output<S>
where
    S: Sink,
    R: Subscriber + for<'a> LookupSpan<'a>,
{
    fn on_new_span(&self, attributes: &Attributes<'_>, id: &Id, context: Context<'_, R>) {
        let Some(_guard) = Emitting::enter() else {
            return;
        };
        let Some(span) = context.span(id) else {
            return;
        };
        let mut extensions = span.extensions_mut();
        if extensions.get_mut::<SpanData>().is_some() {
            return;
        }
        let mut record = Record {
            metadata: attributes.metadata(),
            message: String::new(),
            fields: Map::new(),
        };
        attributes.record(&mut record);
        extensions.insert(SpanData {
            record,
            started: Instant::now(),
        });
    }

    fn on_record(&self, id: &Id, values: &SpanRecord<'_>, context: Context<'_, R>) {
        let Some(_guard) = Emitting::enter() else {
            return;
        };
        let Some(span) = context.span(id) else {
            return;
        };
        if let Some(data) = span.extensions_mut().get_mut::<SpanData>() {
            values.record(&mut data.record);
        }
    }

    fn on_event(&self, event: &Event<'_>, context: Context<'_, R>) {
        let Some(_guard) = Emitting::enter() else {
            return;
        };
        let mut record = Record {
            metadata: event.metadata(),
            message: String::new(),
            fields: Map::new(),
        };
        if let Some(scope) = context.event_scope(event) {
            for span in scope.from_root() {
                if let Some(data) = span.extensions().get::<SpanData>() {
                    record.fields.extend(data.record.fields.clone());
                }
            }
        }
        event.record(&mut record);
        self.0.emit(&record);
    }

    fn on_close(&self, id: Id, context: Context<'_, R>) {
        let Some(_guard) = Emitting::enter() else {
            return;
        };
        let Some(span) = context.span(&id) else {
            return;
        };
        if !self.0.enabled(span.metadata()) {
            return;
        }
        let mut record = Record {
            metadata: span.metadata(),
            message: "span closed".into(),
            fields: Map::new(),
        };
        for ancestor in span.scope().from_root() {
            if let Some(data) = ancestor.extensions().get::<SpanData>() {
                record.fields.extend(data.record.fields.clone());
            }
        }
        let extensions = span.extensions();
        let Some(data) = extensions.get::<SpanData>() else {
            return;
        };
        record.fields.insert("span_name".into(), span.name().into());
        record.fields.insert(
            "duration_ms".into(),
            (data.started.elapsed().as_secs_f64() * 1000.0).into(),
        );
        self.0.emit(&record);
    }
}
