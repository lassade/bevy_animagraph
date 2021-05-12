use bevy::utils::{HashMap, Uuid};

#[derive(Debug, Copy, Clone)]
pub enum Value {
    Float(f32),
    Bool(bool),
}

impl Value {
    pub fn as_float(&self) -> f32 {
        if let Value::Float(value) = self {
            return *value;
        } else {
            0.0
        }
    }

    pub fn as_bool(&self) -> bool {
        if let Value::Bool(value) = self {
            return *value;
        } else {
            false
        }
    }
}

#[derive(Default, Debug)]
pub struct State {
    variables: HashMap<String, Value>,
    time: HashMap<Uuid, f32>,
}

impl State {
    #[inline]
    pub fn get(&self, name: &str) -> Value {
        self.variables
            .get(name)
            .copied()
            .unwrap_or(Value::Bool(false))
    }

    pub fn set(&mut self, name: &str, value: Value) {
        match (self.variables.get_mut(name), value) {
            (Some(Value::Float(y)), Value::Float(x)) => *y = x,
            (Some(Value::Bool(y)), Value::Bool(x)) => *y = x,
            _ => {}
        }
    }

    pub fn entry<K>(&mut self, name: K, default: Value) -> Value
    where
        K: Into<String> + AsRef<str>,
    {
        if let Some(value) = self.variables.get(name.as_ref()) {
            *value
        } else {
            self.variables.insert(name.into(), default);
            default
        }
    }

    #[inline]
    pub fn time(&mut self, uuid: Uuid) -> &mut f32 {
        self.time.entry(uuid).or_insert(0.0)
    }

    #[inline]
    pub fn clear(&mut self) {
        self.variables.clear();
        self.time.clear();
    }
}
