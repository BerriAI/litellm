use std::fmt::{self, Write};
use std::ops::{Deref, DerefMut};
use std::sync::Arc;

use crate::encode::{Encode, IsNull};
use crate::error::Error;
use crate::types::Type;
use crate::{PgConnection, PgTypeInfo, Postgres};

pub(crate) use sqlx_core::arguments::Arguments;
use sqlx_core::error::BoxDynError;

// TODO: buf.patch(|| ...) is a poor name, can we think of a better name? Maybe `buf.lazy(||)` ?
// TODO: Extend the patch system to support dynamic lengths
//       Considerations:
//          - The prefixed-len offset needs to be back-tracked and updated
//          - message::Bind needs to take a &PgArguments and use a `write` method instead of
//            referencing a buffer directly
//          - The basic idea is that we write bytes for the buffer until we get somewhere
//            that has a patch, we then apply the patch which should write to &mut Vec<u8>,
//            backtrack and update the prefixed-len, then write until the next patch offset

#[derive(Default, Debug, Clone)]
pub struct PgArgumentBuffer {
    buffer: Vec<u8>,

    // Number of arguments
    count: usize,

    // Whenever an `Encode` impl needs to defer some work until after we resolve parameter types
    // it can use `patch`.
    //
    // This currently is only setup to be useful if there is a *fixed-size* slot that needs to be
    // tweaked from the input type. However, that's the only use case we currently have.
    patches: Vec<Patch>,

    // Whenever an `Encode` impl encounters a `PgTypeInfo` object that does not have an OID
    // It pushes a "hole" that must be patched later.
    //
    // The hole is a `usize` offset into the buffer with the type name that should be resolved
    // This is done for Records and Arrays as the OID is needed well before we are in an async
    // function and can just ask postgres.
    //
    hole_offsets: Vec<usize>,
    // Separate vecator so that we don't have to generify or duplicate the logic in
    // `PgConnection::resolve_types()`.
    hole_types: Vec<PgTypeInfo>,
}

#[derive(Clone)]
struct Patch {
    buf_offset: usize,
    arg_index: usize,
    #[allow(clippy::type_complexity)]
    callback: Arc<dyn Fn(&mut [u8], &PgTypeInfo) + 'static + Send + Sync>,
}

impl fmt::Debug for Patch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Patch")
            .field("buf_offset", &self.buf_offset)
            .field("arg_index", &self.arg_index)
            .field("callback", &"<callback>")
            .finish()
    }
}

/// Implementation of [`Arguments`] for PostgreSQL.
#[derive(Default, Debug, Clone)]
pub struct PgArguments {
    // Types of each bind parameter
    pub(crate) types: Vec<PgTypeInfo>,

    // Buffer of encoded bind parameters
    pub(crate) buffer: PgArgumentBuffer,
}

impl PgArguments {
    pub(crate) fn add<'q, T>(&mut self, value: T) -> Result<(), BoxDynError>
    where
        T: Encode<'q, Postgres> + Type<Postgres>,
    {
        let type_info = value.produces().unwrap_or_else(T::type_info);

        let buffer_snapshot = self.buffer.snapshot();

        // encode the value into our buffer
        if let Err(error) = self.buffer.encode(value) {
            // reset the value buffer to its previous value if encoding failed,
            // so we don't leave a half-encoded value behind
            self.buffer.reset_to_snapshot(buffer_snapshot);
            return Err(error);
        };

        // remember the type information for this value
        self.types.push(type_info);
        // increment the number of arguments we are tracking
        self.buffer.count += 1;

        Ok(())
    }

    // Apply patches
    // This should only go out and ask postgres if we have not seen the type name yet
    pub(crate) async fn apply_patches(
        &mut self,
        conn: &mut PgConnection,
        parameters: &[PgTypeInfo],
    ) -> Result<(), Error> {
        let PgArgumentBuffer {
            ref patches,
            ref hole_types,
            ref hole_offsets,
            ref mut buffer,
            ..
        } = self.buffer;

        for patch in patches {
            let buf = &mut buffer[patch.buf_offset..];
            let ty = &parameters[patch.arg_index];

            (patch.callback)(buf, ty);
        }

        let resolved_holes = conn.resolve_types(hole_types).await?;

        for (&offset, oid) in hole_offsets.iter().zip(resolved_holes) {
            buffer[offset..][..4].copy_from_slice(&oid.0.to_be_bytes());
        }

        Ok(())
    }
}

impl Arguments for PgArguments {
    type Database = Postgres;

    fn reserve(&mut self, additional: usize, size: usize) {
        self.types.reserve(additional);
        self.buffer.reserve(size);
    }

    fn add<'t, T>(&mut self, value: T) -> Result<(), BoxDynError>
    where
        T: Encode<'t, Self::Database> + Type<Self::Database>,
    {
        self.add(value)
    }

    fn format_placeholder<W: Write>(&self, writer: &mut W) -> fmt::Result {
        write!(writer, "${}", self.buffer.count)
    }

    #[inline(always)]
    fn len(&self) -> usize {
        self.buffer.count
    }
}

impl PgArgumentBuffer {
    pub(crate) fn encode<'q, T>(&mut self, value: T) -> Result<(), BoxDynError>
    where
        T: Encode<'q, Postgres>,
    {
        // Won't catch everything but is a good sanity check
        value_size_int4_checked(value.size_hint())?;

        // reserve space to write the prefixed length of the value
        let offset = self.len();

        self.extend(&[0; 4]);

        // encode the value into our buffer
        let len = if let IsNull::No = value.encode(self)? {
            // Ensure that the value size does not overflow i32
            value_size_int4_checked(self.len() - offset - 4)?
        } else {
            // Write a -1 to indicate NULL
            // NOTE: It is illegal for [encode] to write any data
            debug_assert_eq!(self.len(), offset + 4);
            -1_i32
        };

        // write the len to the beginning of the value
        // (offset + 4) cannot overflow because it would have failed at `self.extend()`.
        self[offset..(offset + 4)].copy_from_slice(&len.to_be_bytes());

        Ok(())
    }

    // Adds a callback to be invoked later when we know the parameter type
    #[cfg_attr(not(feature = "json"), expect(dead_code))]
    pub(crate) fn patch_with<F>(&mut self, callback: F)
    where
        F: Fn(&mut [u8], &PgTypeInfo) + 'static + Send + Sync,
    {
        let offset = self.len();
        let arg_index = self.count;

        self.patches.push(Patch {
            buf_offset: offset,
            arg_index,
            callback: Arc::new(callback),
        });
    }

    // Extends the inner buffer by enough space to have an OID
    // Remembers where the OID goes and type name for the OID
    pub(crate) fn push_hole(&mut self, type_info: PgTypeInfo) {
        let offset = self.len();

        self.extend_from_slice(&0_u32.to_be_bytes());
        self.hole_offsets.push(offset);
        self.hole_types.push(type_info);
    }

    fn snapshot(&self) -> PgArgumentBufferSnapshot {
        let Self {
            buffer,
            count,
            patches,
            hole_offsets,
            ..
        } = self;

        PgArgumentBufferSnapshot {
            buffer_length: buffer.len(),
            count: *count,
            patches_length: patches.len(),
            type_holes_length: hole_offsets.len(),
        }
    }

    fn reset_to_snapshot(
        &mut self,
        PgArgumentBufferSnapshot {
            buffer_length,
            count,
            patches_length,
            type_holes_length,
        }: PgArgumentBufferSnapshot,
    ) {
        self.buffer.truncate(buffer_length);
        self.count = count;
        self.patches.truncate(patches_length);
        self.hole_offsets.truncate(type_holes_length);
        self.hole_types.truncate(type_holes_length);
    }
}

struct PgArgumentBufferSnapshot {
    buffer_length: usize,
    count: usize,
    patches_length: usize,
    type_holes_length: usize,
}

impl Deref for PgArgumentBuffer {
    type Target = Vec<u8>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.buffer
    }
}

impl DerefMut for PgArgumentBuffer {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.buffer
    }
}

pub(crate) fn value_size_int4_checked(size: usize) -> Result<i32, String> {
    i32::try_from(size).map_err(|_| {
        format!(
            "value size would overflow in the binary protocol encoding: {size} > {}",
            i32::MAX
        )
    })
}
