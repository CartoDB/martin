//! One `MapLibre` map drawing still images into its own texture, driven from a single thread.

use std::collections::HashMap;
use std::future::Future;
use std::ops::Deref;
use std::path::Path;
use std::pin::Pin;
use std::sync::{Arc, Condvar, Mutex, PoisonError};
use std::task::{Context, Poll, Wake, Waker};
use std::time::{Duration, Instant};

use image::RgbaImage;
use maplibre_native_ffi::{
    CameraOptions, CameraUpdate, CameraUpdateMode, CommandCompletion, GraphicsDeviceHandle,
    GraphicsDeviceTextureDescriptor, LogicalExtent, MapHandle, MapMode, MapOptions, NativeFuture, RenderDriverKind,
    RenderSessionAttachOptions, RenderSessionHandle, RuntimeEvent, RuntimeEventFlag, RuntimeEventHandler,
    RuntimeEventType, RuntimeHandle, RuntimeOptions, TextureReadbackResult,
};

use tracing::warn;

use crate::resources::styles::StyleError;

/// How long a renderer waits for `MapLibre` to finish any one step before it gives up.
///
/// Far above a cold render (about 3 s at p99 on llvmpipe), so only a render that will never finish hits it.
pub const RENDER_TIMEOUT: Duration = Duration::from_secs(30);

/// A `MapLibre` runtime and graphics device shared by every renderer of a pool, with one ambient cache.
pub struct Runtime {
    /// The native runtime.
    handle: Releasable<RuntimeHandle>,
    /// The device every renderer's texture lives on.
    device: GraphicsDeviceHandle,
    /// Style-load listeners, by the id of the map they listen to.
    listeners: Arc<Mutex<HashMap<u64, Arc<StyleListener>>>>,
    /// How long its renderers wait for each step, [`RENDER_TIMEOUT`] unless a test shortens it.
    timeout: Duration,
}

impl std::fmt::Debug for Runtime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Runtime").finish_non_exhaustive()
    }
}

impl Drop for Runtime {
    /// Releases the runtime and waits until its native threads have stopped.
    fn drop(&mut self) {
        let signal = Arc::new(Signal::default());
        let Some(handle) = self.handle.take() else { return };
        let released = (handle.release())
            .map_err(|(_, err)| StyleError::from(err))
            .and_then(|released| block_on(&signal, released, RENDER_TIMEOUT));
        if let Err(err) = released.and_then(|released| Ok(released?)) {
            warn!(error = %err, "Failed to release the MapLibre runtime");
        }
    }
}

impl Runtime {
    /// Creates a runtime whose ambient cache holds up to `ambient_cache_bytes`
    /// (`None` keeps `MapLibre`'s default, `Some(0)` disables it).
    ///
    /// # Errors
    ///
    /// Returns [`StyleError::GraphicsUnavailable`] when no graphics device can be created.
    pub fn new(ambient_cache_bytes: Option<u64>) -> Result<Self, StyleError> {
        let device = GraphicsDeviceHandle::create(None).map_err(|err| StyleError::GraphicsUnavailable {
            reason: err.to_string(),
        })?;
        let handle = RuntimeHandle::create(&RuntimeOptions::default())?;
        let listeners: Arc<Mutex<HashMap<u64, Arc<StyleListener>>>> = Arc::default();
        let routes = Arc::downgrade(&listeners);
        let signal = Arc::new(Signal::default());
        let handler = RuntimeEventHandler::new(move |batch| {
            let Some(routes) = routes.upgrade() else { return };
            let routes = routes.lock().unwrap_or_else(PoisonError::into_inner);
            for event in batch {
                if let Some(listener) = routes.get(&event.source) {
                    listener.observe(&event);
                }
            }
        });
        block_on(&signal, handle.set_event_handler(handler)?, RENDER_TIMEOUT)??;
        if let Some(bytes) = ambient_cache_bytes {
            block_on(&signal, handle.set_maximum_ambient_cache_size(bytes)?, RENDER_TIMEOUT)??;
        }
        Ok(Self {
            handle: Releasable(Some(handle)),
            device,
            listeners,
            timeout: RENDER_TIMEOUT,
        })
    }

    /// Makes its renderers give up after `timeout` instead of [`RENDER_TIMEOUT`].
    #[cfg(test)]
    pub(crate) fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }
}

/// What a map reported about loading a style.
#[derive(Debug, Clone)]
enum StyleOutcome {
    /// The style loaded.
    Loaded,
    /// The style failed to load, with `MapLibre`'s reason.
    Failed(String),
}

/// Collects the style-load outcomes one map reports while a style load is pending.
///
/// Outcomes are kept with their generation, because only once the style command has
/// committed is it known which generation the new style starts at.
#[derive(Debug, Default)]
struct StyleListener {
    /// Outcomes reported since the pending load began, with their generations,
    /// or `None` while no load is pending.
    outcomes: Mutex<Option<Vec<(u64, StyleOutcome)>>>,
    /// Raised whenever an outcome is recorded.
    signal: Arc<Signal>,
}

impl StyleListener {
    /// Records `event` if it is a style-load outcome and a load is pending.
    fn observe(&self, event: &RuntimeEvent) {
        let outcome = if matches!(event.r#type, RuntimeEventType::MapStyleLoaded) {
            StyleOutcome::Loaded
        } else if matches!(event.r#type, RuntimeEventType::MapLoadingFailed) {
            StyleOutcome::Failed(event.message.clone())
        } else {
            return;
        };
        if let Some(outcomes) = self.outcomes.lock().unwrap_or_else(PoisonError::into_inner).as_mut() {
            outcomes.push((event.generation, outcome));
            self.signal.raise();
        }
    }

    /// Starts collecting outcomes for a new load, dropping any earlier ones.
    fn begin(&self) {
        *self.outcomes.lock().unwrap_or_else(PoisonError::into_inner) = Some(Vec::new());
    }

    /// Waits up to `timeout` for the first outcome at or after `generation`, and stops collecting.
    fn wait(&self, generation: u64, timeout: Duration) -> Result<StyleOutcome, StyleError> {
        let deadline = Instant::now() + timeout;
        loop {
            let seen = self.signal.seen();
            let mut outcomes = self.outcomes.lock().unwrap_or_else(PoisonError::into_inner);
            let found = outcomes
                .iter()
                .flatten()
                .find(|(reported, _)| *reported >= generation)
                .map(|(_, outcome)| outcome.clone());
            if let Some(outcome) = found {
                *outcomes = None;
                return Ok(outcome);
            }
            drop(outcomes);
            if !self.signal.wait_past(seen, Some(deadline)) {
                *self.outcomes.lock().unwrap_or_else(PoisonError::into_inner) = None;
                return Err(StyleError::RenderTimeout { after: timeout });
            }
        }
    }
}

/// A map in tile or static mode with a session-owned texture it reads its still images back from.
///
/// Every call blocks the calling thread until `MapLibre` answers.
#[derive(Debug)]
pub struct Renderer {
    /// The pool's runtime, kept alive for as long as the map.
    runtime: Arc<Runtime>,
    /// The map the camera, style and overlays apply to.
    map: Releasable<MapHandle>,
    /// The render target the map draws into.
    session: Releasable<RenderSessionHandle>,
    /// Style-load outcomes of `map`.
    style: Arc<StyleListener>,
    /// Raised by every native completion this renderer waits on.
    signal: Arc<Signal>,
}

impl Renderer {
    /// Creates a map of `mode` drawing `width` x `height` logical pixels at `pixel_ratio`.
    pub fn new(
        runtime: &Arc<Runtime>,
        mode: MapMode,
        width: u32,
        height: u32,
        pixel_ratio: f64,
    ) -> Result<Self, StyleError> {
        let signal = Arc::new(Signal::default());
        let extent = LogicalExtent::new(width, height, pixel_ratio);
        let options = MapOptions {
            map_mode: mode,
            event_mask: Some(RuntimeEventFlag::MAP_STYLE_LOADED | RuntimeEventFlag::MAP_LOADING_FAILED),
            ..MapOptions::new(extent)
        };
        let (map, created) = runtime.handle.create_map(&options)?;
        let style = Arc::new(StyleListener::default());
        runtime
            .listeners
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .insert(map.id(), Arc::clone(&style));
        block_on(&signal, created, runtime.timeout)??;

        let attach = RenderSessionAttachOptions::new(RenderDriverKind::CoreWorker);
        let descriptor = GraphicsDeviceTextureDescriptor::new(extent);
        let (session, attached) = map.attach_graphics_device_texture(&runtime.device, &descriptor, &attach)?;
        let renderer = Self {
            runtime: Arc::clone(runtime),
            map: Releasable(Some(map)),
            session: Releasable(Some(session)),
            style,
            signal,
        };
        renderer.wait(attached)??;
        Ok(renderer)
    }

    /// Loads the style at `path` and waits until `MapLibre` has loaded or rejected it.
    pub fn load_style(&mut self, path: &Path) -> Result<(), StyleError> {
        if !path.is_file() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("Path {} is not a file", path.display()),
            )
            .into());
        }
        let Some(path) = path.to_str() else {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Path {} is not valid UTF-8", path.display()),
            )
            .into());
        };
        let url = format!("file://{path}");
        self.set_style(|map| map.set_style_url(&url))
    }

    /// Loads the style document `json` and waits until `MapLibre` has loaded or rejected it.
    pub fn load_style_json(&mut self, json: &str) -> Result<(), StyleError> {
        self.set_style(|map| map.set_style_json(json))
    }

    /// Sends the style command `set` and waits until `MapLibre` has loaded or rejected its style.
    fn set_style(
        &self,
        set: impl FnOnce(&MapHandle) -> maplibre_native_ffi::Result<NativeFuture<CommandCompletion>>,
    ) -> Result<(), StyleError> {
        self.style.begin();
        // A failed style command loads nothing, so no outcome would ever arrive.
        let set = self.wait(set(&self.map)?)?.map_err(|err| StyleError::StyleLoadError {
            message: err.to_string(),
        })?;
        match self.style.wait(set.generation, self.runtime.timeout)? {
            StyleOutcome::Loaded => Ok(()),
            StyleOutcome::Failed(message) => Err(StyleError::StyleLoadError { message }),
        }
    }

    /// Resizes the drawing to `width` x `height` logical pixels, keeping the map and its style.
    pub fn resize(&mut self, width: u32, height: u32) -> Result<(), StyleError> {
        let scale_factor = self.session.get_snapshot()?.extent.scale_factor;
        self.wait(self.session.resize(LogicalExtent::new(width, height, scale_factor))?)??;
        Ok(())
    }

    /// The map, for adding and removing overlay sources and layers.
    pub fn map(&self) -> &MapHandle {
        &self.map
    }

    /// Waits for `future`, failing once the runtime's timeout has passed.
    fn wait<F: Future + Unpin>(&self, future: F) -> Result<F::Output, StyleError> {
        block_on(&self.signal, future, self.runtime.timeout)
    }

    /// Waits for the outcome of a style command, failing if `MapLibre` rejected it.
    ///
    /// Not bounded by the timeout: a style command touches no network.
    pub fn command(
        &self,
        command: maplibre_native_ffi::Result<NativeFuture<CommandCompletion>>,
    ) -> Result<(), maplibre_native_ffi::Error> {
        let Some(completion) = poll_until(&self.signal, command?, None) else {
            unreachable!("only a deadline stops the wait")
        };
        completion?;
        Ok(())
    }

    /// Moves the camera to `camera` and renders the still image `MapLibre` produces once everything it shows has loaded.
    pub fn render(&mut self, camera: CameraOptions) -> Result<RgbaImage, StyleError> {
        self.map.discarding().update_camera(&CameraUpdate {
            mode: CameraUpdateMode::Jump,
            camera,
            ..CameraUpdate::default()
        })?;
        self.wait(self.map.request_still_image()?)??;
        // Frame results queue up on the session until drained, and the still image needs none of them.
        self.session.drain_frame_results()?;
        let readback = self.wait(self.session.read_texture()?)??;
        Ok(unpremultiply(readback))
    }
}

impl Drop for Renderer {
    /// Detaches and releases the session, then the map, waiting for each, so no native
    /// worker still draws when the runtime or the process goes away.
    fn drop(&mut self) {
        let detached = (self.session.detach())
            .map_err(StyleError::from)
            .and_then(|detached| self.wait(detached));
        if let Err(err) = detached.and_then(|detached| Ok(detached?)) {
            warn!(error = %err, "Failed to detach a render session");
        }
        if let Some(session) = self.session.take() {
            let released = (session.release())
                .map_err(|(_, err)| StyleError::from(err))
                .and_then(|released| self.wait(released));
            if let Err(err) = released.and_then(|released| Ok(released?)) {
                warn!(error = %err, "Failed to release a render session");
            }
        }
        let Some(map) = self.map.take() else { return };
        let id = map.id();
        let released = (map.release())
            .map_err(|(_, err)| StyleError::from(err))
            .and_then(|released| self.wait(released));
        if let Err(err) = released.and_then(|released| Ok(released?)) {
            warn!(error = %err, "Failed to release a map");
        }
        self.runtime
            .listeners
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .remove(&id);
    }
}

/// A native handle that `Drop` moves out to release it and wait for its teardown.
#[derive(Debug)]
struct Releasable<T>(Option<T>);

impl<T> Releasable<T> {
    /// Moves the handle out, leaving nothing to release twice.
    fn take(&mut self) -> Option<T> {
        self.0.take()
    }
}

impl<T> Deref for Releasable<T> {
    type Target = T;

    fn deref(&self) -> &T {
        self.0
            .as_ref()
            .expect("a handle is only moved out while its owner drops")
    }
}

/// Converts `MapLibre`'s premultiplied readback to straight alpha, rounding as `MapLibre`'s own
/// `unpremultiply` does, and drops any row padding.
fn unpremultiply(readback: TextureReadbackResult) -> RgbaImage {
    let TextureReadbackResult { mut data, info } = readback;
    let row = info.width as usize * 4;
    let stride = info.stride as usize;
    if stride != row {
        for y in 1..info.height as usize {
            data.copy_within(y * stride..y * stride + row, y * row);
        }
    }
    data.truncate(row * info.height as usize);
    for pixel in data.as_chunks_mut::<4>().0 {
        let alpha = u16::from(pixel[3]);
        if alpha != 0 && alpha != 255 {
            for channel in &mut pixel[..3] {
                // At most 255, since a premultiplied channel never exceeds its alpha.
                *channel = u8::try_from((255 * u16::from(*channel) + alpha / 2) / alpha).unwrap_or(u8::MAX);
            }
        }
    }
    RgbaImage::from_raw(info.width, info.height, data).expect("readback holds width x height RGBA pixels")
}

/// A counter that threads wait on to move past, raised by native callbacks and future wakers.
#[derive(Debug, Default)]
struct Signal {
    /// How often the signal was raised.
    raised: Mutex<u64>,
    /// Notified on every raise.
    changed: Condvar,
}

impl Signal {
    /// Records one more raise and wakes the waiting thread.
    fn raise(&self) {
        *self.raised.lock().unwrap_or_else(PoisonError::into_inner) += 1;
        self.changed.notify_all();
    }

    /// The raise count, to pass to [`Signal::wait_past`] after checking what the raise announces.
    fn seen(&self) -> u64 {
        *self.raised.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Blocks until the signal was raised after `seen`, returning `false` if `deadline` passed first.
    fn wait_past(&self, seen: u64, deadline: Option<Instant>) -> bool {
        let raised = self.raised.lock().unwrap_or_else(PoisonError::into_inner);
        let Some(deadline) = deadline else {
            drop(
                self.changed
                    .wait_while(raised, |raised| *raised == seen)
                    .unwrap_or_else(PoisonError::into_inner),
            );
            return true;
        };
        let timeout = deadline.saturating_duration_since(Instant::now());
        let (raised, result) = self
            .changed
            .wait_timeout_while(raised, timeout, |raised| *raised == seen)
            .unwrap_or_else(PoisonError::into_inner);
        drop(raised);
        !result.timed_out()
    }
}

impl Wake for Signal {
    fn wake(self: Arc<Self>) {
        self.raise();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.raise();
    }
}

/// Blocks the calling thread until `future` resolves, sleeping on `signal` between polls,
/// and fails once `timeout` has passed.
///
/// A future left behind still resolves natively; nothing awaits it any more.
fn block_on<F: Future + Unpin>(signal: &Arc<Signal>, future: F, timeout: Duration) -> Result<F::Output, StyleError> {
    poll_until(signal, future, Some(Instant::now() + timeout)).ok_or(StyleError::RenderTimeout { after: timeout })
}

/// Polls `future` until it resolves, sleeping on `signal` between polls, or until `deadline` passes.
fn poll_until<F: Future + Unpin>(signal: &Arc<Signal>, mut future: F, deadline: Option<Instant>) -> Option<F::Output> {
    let waker = Waker::from(Arc::clone(signal));
    let mut context = Context::from_waker(&waker);
    loop {
        let seen = signal.seen();
        if let Poll::Ready(output) = Pin::new(&mut future).poll(&mut context) {
            return Some(output);
        }
        if !signal.wait_past(seen, deadline) {
            return None;
        }
    }
}

#[cfg(test)]
mod tests {
    use std::future;

    use super::*;

    #[test]
    fn a_wait_fails_once_its_timeout_passes() {
        let signal = Arc::new(Signal::default());
        let timeout = Duration::from_millis(50);

        let started = Instant::now();
        let result = block_on(&signal, future::pending::<()>(), timeout);

        assert!(
            matches!(result, Err(StyleError::RenderTimeout { after }) if after == timeout),
            "{result:?}"
        );
        assert!(started.elapsed() >= timeout);
        assert!(matches!(block_on(&signal, future::ready(7), timeout), Ok(7)));
    }
}
