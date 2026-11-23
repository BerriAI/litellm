use std::{
    collections::{HashMap, HashSet},
    future::Future,
    sync::{
        Arc, Mutex, OnceLock, PoisonError,
        atomic::{AtomicU64, Ordering},
    },
};

use futures_util::future::BoxFuture;
use litellm_callbacks::{PayloadHost, PayloadRef, RenderOptions, Rendered};
use pyo3::{
    prelude::*,
    types::{PyBool, PyCFunction, PyDict, PyFloat, PyInt, PyList, PySet, PyString, PyTuple},
};
use tokio::sync::oneshot;

use litellm_python_compat::json::{float_text, write_string};

tokio::task_local! {
    static RENDER_LOOP: Option<Arc<Py<PyAny>>>;
}

#[derive(Default)]
struct PythonPayloads {
    stashed: Mutex<HashMap<u64, Py<PyAny>>>,
    next: AtomicU64,
}

fn payloads() -> &'static Arc<PythonPayloads> {
    static PAYLOADS: OnceLock<Arc<PythonPayloads>> = OnceLock::new();
    PAYLOADS.get_or_init(Arc::default)
}

pub fn host() -> Arc<dyn PayloadHost> {
    Arc::clone(payloads()) as Arc<dyn PayloadHost>
}

pub fn stash(payload: Py<PyAny>) -> PayloadRef {
    let payloads = payloads();
    let handle = payloads.next.fetch_add(1, Ordering::Relaxed);
    payloads
        .stashed
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .insert(handle, payload);
    PayloadRef::new(handle, host())
}

pub fn running_loop(py: Python<'_>) -> Option<Arc<Py<PyAny>>> {
    py.import("asyncio")
        .and_then(|asyncio| asyncio.call_method0("get_running_loop"))
        .ok()
        .map(|event_loop| Arc::new(event_loop.unbind()))
}

pub async fn on_render_loop<F: Future>(event_loop: Option<Arc<Py<PyAny>>>, run: F) -> F::Output {
    RENDER_LOOP.scope(event_loop, run).await
}

fn panic_message(panic: &(dyn std::any::Any + Send)) -> String {
    panic
        .downcast_ref::<&str>()
        .map(|message| (*message).to_owned())
        .or_else(|| panic.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "a panic without a message".to_owned())
}

impl PythonPayloads {
    fn get(&self, py: Python<'_>, handle: u64) -> Result<Py<PyAny>, String> {
        self.stashed
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .get(&handle)
            .map(|payload| payload.clone_ref(py))
            .ok_or_else(|| "the payload was already released".to_owned())
    }

    fn encode_all(&self, py: Python<'_>, handles: &[u64], options: &RenderOptions) -> Rendered {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            handles
                .iter()
                .map(|handle| {
                    let payload = self.get(py, *handle)?;
                    encode_one(payload.bind(py), options).map_err(|error| error.to_string())
                })
                .collect()
        }))
        .unwrap_or_else(|panic| Err(panic_message(panic.as_ref())))
    }
}

fn with_null_keys<'py>(
    payload: &Bound<'py, PyAny>,
    options: &RenderOptions,
) -> PyResult<Bound<'py, PyAny>> {
    let Ok(entries) = payload.cast::<PyDict>() else {
        return Ok(payload.clone());
    };
    if options.null_keys.is_empty() {
        return Ok(payload.clone());
    }
    let copied = entries.copy()?;
    for key in &options.null_keys {
        copied.set_item(key, payload.py().None())?;
    }
    Ok(copied.into_any())
}

fn encode_one(payload: &Bound<'_, PyAny>, options: &RenderOptions) -> PyResult<Vec<u8>> {
    let payload = with_null_keys(payload, options)?;
    let mut out = Vec::new();
    Encoder::new(payload.py())?.encode(&payload, 0, &mut out)?;
    Ok(out)
}

fn render_on_loop(
    py: Python<'_>,
    event_loop: &Py<PyAny>,
    payloads: Arc<PythonPayloads>,
    handles: Vec<u64>,
    options: RenderOptions,
    reply: oneshot::Sender<Rendered>,
) -> PyResult<()> {
    let reply = Mutex::new(Some(reply));
    let render = PyCFunction::new_closure(
        py,
        None,
        None,
        move |args: &Bound<'_, PyTuple>, _: Option<&Bound<'_, PyDict>>| {
            let rendered = payloads.encode_all(args.py(), &handles, &options);
            if let Some(reply) = reply.lock().unwrap_or_else(PoisonError::into_inner).take() {
                drop(reply.send(rendered));
            }
        },
    )?;
    event_loop
        .bind(py)
        .call_method1("call_soon_threadsafe", (render,))?;
    Ok(())
}

impl PayloadHost for PythonPayloads {
    fn render(&self, handles: Vec<u64>, options: RenderOptions) -> BoxFuture<'static, Rendered> {
        let payloads = Arc::clone(payloads());
        let event_loop = RENDER_LOOP.try_with(Clone::clone).ok().flatten();
        Box::pin(async move {
            let Some(event_loop) = event_loop else {
                return tokio::task::spawn_blocking(move || {
                    Python::attach(|py| payloads.encode_all(py, &handles, &options))
                })
                .await
                .map_err(|error| error.to_string())?;
            };
            let (reply, rendered) = oneshot::channel();
            tokio::task::spawn_blocking(move || {
                Python::attach(|py| {
                    render_on_loop(py, &event_loop, payloads, handles, options, reply)
                        .map_err(|error| error.to_string())
                })
            })
            .await
            .map_err(|error| error.to_string())??;
            rendered
                .await
                .map_err(|_| "the event loop closed before the payload was rendered".to_owned())?
        })
    }

    fn release(&self, handle: u64) {
        let released = self
            .stashed
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .remove(&handle);
        drop(released);
    }
}

struct Encoder<'py> {
    base_model: Bound<'py, PyAny>,
    safe_json_structure: Bound<'py, PyAny>,
    json_dumps: Bound<'py, PyAny>,
    int_repr: Bound<'py, PyAny>,
    max_depth: usize,
    seen: HashSet<usize>,
}

impl<'py> Encoder<'py> {
    fn new(py: Python<'py>) -> PyResult<Self> {
        Ok(Self {
            base_model: py.import("pydantic")?.getattr("BaseModel")?,
            safe_json_structure: py
                .import("litellm.litellm_core_utils.safe_json_dumps")?
                .getattr("safe_json_structure")?,
            json_dumps: py.import("json")?.getattr("dumps")?,
            int_repr: py.get_type::<PyInt>().getattr("__repr__")?,
            max_depth: py
                .import("litellm.constants")?
                .getattr("DEFAULT_MAX_RECURSE_DEPTH")?
                .extract()?,
            seen: HashSet::new(),
        })
    }

    fn raw(&self, value: &Bound<'py, PyAny>, out: &mut Vec<u8>) -> PyResult<()> {
        let text: String = self.json_dumps.call1((value,))?.extract()?;
        out.extend_from_slice(text.as_bytes());
        Ok(())
    }

    fn text(&self, text: &Bound<'py, PyString>, out: &mut Vec<u8>) -> PyResult<()> {
        match text.to_str() {
            Ok(text) if text.contains('\0') => write_string(out, &text.replace('\0', "")),
            Ok(text) => write_string(out, text),
            Err(_) => self.raw(&text.call_method1("replace", ("\0", ""))?, out)?,
        }
        Ok(())
    }

    fn entries(
        &mut self,
        items: impl Iterator<Item = (Bound<'py, PyAny>, Bound<'py, PyAny>)>,
        depth: usize,
        out: &mut Vec<u8>,
    ) -> PyResult<()> {
        let mut written: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
        for (key, value) in items {
            let Ok(key) = key.cast_into::<PyString>() else {
                continue;
            };
            let mut encoded_key = Vec::new();
            self.text(&key, &mut encoded_key)?;
            let mut encoded_value = Vec::new();
            self.encode(&value, depth + 1, &mut encoded_value)?;
            match written
                .iter_mut()
                .find(|(present, _)| *present == encoded_key)
            {
                Some((_, slot)) => *slot = encoded_value,
                None => written.push((encoded_key, encoded_value)),
            }
        }
        out.push(b'{');
        for (index, (key, value)) in written.iter().enumerate() {
            if index > 0 {
                out.extend_from_slice(b", ");
            }
            out.extend_from_slice(key);
            out.extend_from_slice(b": ");
            out.extend_from_slice(value);
        }
        out.push(b'}');
        Ok(())
    }

    fn items(
        &mut self,
        items: impl Iterator<Item = Bound<'py, PyAny>>,
        depth: usize,
        out: &mut Vec<u8>,
    ) -> PyResult<()> {
        out.push(b'[');
        for (index, item) in items.enumerate() {
            if index > 0 {
                out.extend_from_slice(b", ");
            }
            self.encode(&item, depth + 1, out)?;
        }
        out.push(b']');
        Ok(())
    }

    fn scalar(&self, value: &Bound<'py, PyAny>, out: &mut Vec<u8>) -> PyResult<bool> {
        if let Ok(text) = value.cast::<PyString>() {
            self.text(text, out)?;
        } else if value.is_none() {
            out.extend_from_slice(b"null");
        } else if let Ok(flag) = value.cast::<PyBool>() {
            out.extend_from_slice(if flag.is_true() { b"true" } else { b"false" });
        } else if let Ok(integer) = value.cast::<PyInt>() {
            match integer.extract::<i64>() {
                Ok(integer) => out.extend_from_slice(integer.to_string().as_bytes()),
                Err(_) => {
                    let text: String = self.int_repr.call1((value,))?.extract()?;
                    out.extend_from_slice(text.as_bytes());
                }
            }
        } else if let Ok(float) = value.cast::<PyFloat>() {
            out.extend_from_slice(float_text(float.value()).as_bytes());
        } else {
            return Ok(false);
        }
        Ok(true)
    }

    fn encode(
        &mut self,
        value: &Bound<'py, PyAny>,
        depth: usize,
        out: &mut Vec<u8>,
    ) -> PyResult<()> {
        if depth > self.max_depth {
            write_string(out, "MaxDepthExceeded");
            return Ok(());
        }
        if self.scalar(value, out)? {
            return Ok(());
        }
        let identity = value.as_ptr() as usize;
        if self.seen.contains(&identity) {
            write_string(out, "CircularReference Detected");
            return Ok(());
        }
        if value.cast::<PySet>().is_ok() {
            let structure = self
                .safe_json_structure
                .call1((value, self.max_depth - depth))?;
            return self.raw(&structure, out);
        }
        self.seen.insert(identity);
        let encoded = if let Ok(dict) = value.cast::<PyDict>() {
            self.entries(dict.iter(), depth, out)
        } else if let Ok(list) = value.cast::<PyList>() {
            self.items(list.iter(), depth, out)
        } else if let Ok(tuple) = value.cast::<PyTuple>() {
            self.items(tuple.iter(), depth, out)
        } else if value.is_instance(&self.base_model)? {
            let dumped = value.call_method0("model_dump")?;
            self.encode(&dumped, depth + 1, out)
        } else {
            match value.str() {
                Ok(text) => self.text(&text, out)?,
                Err(_) => write_string(out, "Unserializable Object"),
            }
            return Ok(());
        };
        self.seen.remove(&identity);
        encoded
    }
}
