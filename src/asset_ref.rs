use std::{cell::Cell, ops::Deref, ptr::NonNull};

use bevy::asset::{Asset, AssetServer, Handle};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Serializable Handle
#[derive(Default, Debug)]
pub struct AssetRef<T: Asset>(Handle<T>);

impl<T: Asset> Clone for AssetRef<T> {
    #[inline]
    fn clone(&self) -> AssetRef<T> {
        AssetRef(self.0.clone())
    }
}

impl<T: Asset> Deref for AssetRef<T> {
    type Target = Handle<T>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T: Asset> AsRef<Handle<T>> for AssetRef<T> {
    fn as_ref(&self) -> &Handle<T> {
        &self.0
    }
}

impl<T: Asset> From<Handle<T>> for AssetRef<T> {
    fn from(handle: Handle<T>) -> Self {
        AssetRef(handle)
    }
}

impl<T: Asset> From<AssetRef<T>> for Handle<T> {
    fn from(labeled: AssetRef<T>) -> Self {
        labeled.0
    }
}

impl<T: Asset> Serialize for AssetRef<T> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let path = ASSET_SERVER.with(|server| {
            server.get().and_then(|ptr| {
                // SAFETY: The thread local [`ASSET_SERVER`] can only
                // be set by a valid [`AssetServer`] instance that will
                // also make sure to set it to [`None`] once it's no longer is valid
                let server = unsafe { ptr.as_ref() };
                server.get_handle_path(&self.0).map(|asset_path| {
                    let mut path = asset_path.path().to_string_lossy().to_string();
                    if let Some(label) = asset_path.label() {
                        path.push('#');
                        path.push_str(label);
                    }
                    path
                })
            })
        });

        path.serialize(serializer)
    }
}

impl<'de, T: Asset> Deserialize<'de> for AssetRef<T> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Ok(AssetRef(
            Option::<String>::deserialize(deserializer)?
                .and_then(|path| {
                    ASSET_SERVER.with(|server| {
                        server.get().map(|ptr| {
                            // SAFETY: The thread local [`ASSET_SERVER`] can only
                            // be set by a valid [`AssetServer`] instance that will
                            // also make sure to set it to [`None`] once it's no longer is valid
                            let server = unsafe { ptr.as_ref() };
                            server.load(path.as_str())
                        })
                    })
                })
                .unwrap_or_default(),
        ))
    }
}

thread_local! {
    static ASSET_SERVER: Cell<Option<NonNull<AssetServer>>> = Cell::new(None);
}

pub trait AssetSerializer {
    fn serialize_with_asset_refs<S, T>(&self, serializer: S, value: &T) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
        T: Serialize;

    fn deserialize_with_asset_refs<'de, D, T>(&self, deserializer: D) -> Result<T, D::Error>
    where
        D: serde::Deserializer<'de>,
        T: serde::Deserialize<'de>;
}

impl AssetSerializer for AssetServer {
    fn serialize_with_asset_refs<S, T>(&self, serializer: S, value: &T) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
        T: Serialize,
    {
        ASSET_SERVER.with(|key| {
            key.replace(NonNull::new(self as *const _ as *mut _));
            let result = value.serialize(serializer);
            key.replace(None);
            result
        })
    }

    fn deserialize_with_asset_refs<'de, D, T>(&self, deserializer: D) -> Result<T, D::Error>
    where
        D: Deserializer<'de>,
        T: Deserialize<'de>,
    {
        ASSET_SERVER.with(|key| {
            key.replace(NonNull::new(self as *const _ as *mut _));
            let result = T::deserialize(deserializer);
            key.replace(None);
            result
        })
    }
}
