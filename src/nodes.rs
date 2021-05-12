use bevy::{animation::Clip, asset::Handle, utils::Uuid};

use crate::{State, Value};

#[derive(Debug)]
pub struct NodeResponse {
    pub clip: Handle<Clip>,
    pub weight: f32,
    pub time: f32,
}

pub trait Node {
    //fn visit

    fn uuid(&self) -> Uuid;

    fn evaluate(&self, delta_time: f32, state: &mut State, response: &mut Vec<NodeResponse>)
        -> f32;
}

pub struct FloatSwitch {
    uuid: Uuid,
    pub parameter: String,
    pub time_scale: f32,
    pub a: Box<dyn Node>,
    pub b: Box<dyn Node>,
}

impl Node for FloatSwitch {
    fn uuid(&self) -> Uuid {
        self.uuid
    }

    fn evaluate(&self, state: &mut State, response: &mut Vec<NodeResponse>) -> f32 {
        let value = state.entry(&self.parameter, Value::Float(0.0)).as_float();
        float_evaluate(&self.uuid, value, &*self.a, &*self.b, state, response)
    }
}

#[inline]
pub(crate) fn float_evaluate(
    uuid: &Uuid,
    value: f32,
    a: &dyn Node,
    b: &dyn Node,
    state: &mut State,
    response: &mut Vec<NodeResponse>,
) -> f32 {
    let v1 = value.clamp(0.0, 1.0);
    let v0 = 1.0 - v1;

    let i0 = response.len();
    let t0 = a.evaluate(state, response);
    let i1 = response.len();
    let t1 = b.evaluate(state, response);

    // Match clips normalized time
    let time = t0 * v0 + t1 * v1;
    let n = 1.0 / time;
    let ts0 = t0 * n;
    let ts1 = t1 * n;

    response.iter_mut().skip(i0).take(i1 - i0).for_each(|r| {
        r.weight *= v0;
        r.time_scale *= ts0;
    });
    response.iter_mut().skip(i1).for_each(|r| {
        r.weight *= v1;
        r.time_scale *= ts1;
    });

    time
}

pub struct BoolSwitch {
    uuid: Uuid,
    pub name: String,
    pub on: Box<dyn Node>,
    pub off: Box<dyn Node>,
}

impl Node for BoolSwitch {
    fn uuid(&self) -> Uuid {
        self.uuid
    }

    fn evaluate(&self, state: &mut State, response: &mut Vec<NodeResponse>) -> f32 {
        let value = state.entry(&self.name, Value::Bool(false)).as_bool();
        if value {
            self.on.evaluate(state, response)
        } else {
            self.off.evaluate(state, response)
        }
    }
}

pub struct Linear1DSwitch {
    uuid: Uuid,
    nodes: Vec<(Box<dyn Node>, f32)>,
    pub name: String,
}

impl Node for Linear1DSwitch {
    fn uuid(&self) -> Uuid {
        self.uuid
    }

    fn evaluate(&self, state: &mut State, response: &mut Vec<NodeResponse>) -> f32 {
        let value = state.entry(&self.name, Value::Float(0.0)).as_float();

        if let Some(i) = self.nodes.iter().position(|(_, t)| *t > value) {
            if i == 0 {
                return self.nodes[0].0.evaluate(state, response);
            } else {
                let (off, a) = &self.nodes[i - 1];
                let (on, b) = &self.nodes[i];
                let value = (value - *a) / (*b - *a).max(1e-12);
                return float_evaluate(value, &**off, &**on, state, response);
            }
        } else {
            if let Some((node, _)) = self.nodes.last() {
                return node.evaluate(state, response);
            }
        }

        // Empty
        0.0
    }
}

// #[derive(Debug, Clone)]
// pub struct Linear2DSwitch {
// }

#[derive(Debug)]
pub struct Animation {
    uuid: Uuid,
    duration: f32,
    /// Time parameter
    pub parameter: Option<String>,
    pub time_scale: f32,
    pub warp: bool,
    pub clip: Handle<Clip>,
}

impl Node for Animation {
    fn uuid(&self) -> Uuid {
        self.uuid
    }

    fn evaluate(
        &self,
        delta_time: f32,
        state: &mut State,
        response: &mut Vec<NodeResponse>,
    ) -> f32 {
        if let Some(parameter) = &self.parameter {
            // Time fixed
            let time = state
                .entry(parameter, Value::Float(0.0))
                .as_float()
                .clamp(0.0, 1.0);

            response.push(NodeResponse {
                clip: self.clip.clone(),
                weight: 1.0,
                time,
            });

            self.duration
        } else {
            // Time run
            let duration = self.duration * self.time_scale;
            let n = state.time(self.uuid);
            let mut time = (*n) * duration + delta_time;
            if time > self.duration {
                if self.warp {
                    *n = 0.0;
                    time = 0.0;
                } else {
                    *n = 1.0;
                    time = duration;
                }
            } else {
                *n = time / duration;
            }

            response.push(NodeResponse {
                clip: self.clip.clone(),
                weight: 1.0,
                time,
            });

            duration
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn float_switch() {
        let idle_run = FloatSwitch {
            parameter: "Speed".to_string(),
            a: Box::new(Animation {
                time: 6.0,
                ..Default::default()
            }),
            b: Box::new(Animation {
                time: 1.0,
                ..Default::default()
            }),
        };
        let mut state = State::default();
        let mut response = vec![];

        let time = idle_run.evaluate(&mut state, &mut response);
        assert!((time - 6.0).abs() < std::f32::EPSILON);
        assert!((state.get("Speed").as_float() - 0.0).abs() < std::f32::EPSILON);
        assert_eq!(response.len(), 2);
        assert!((response[0].weight - 1.0).abs() < std::f32::EPSILON);
        assert!((response[1].weight - 0.0).abs() < std::f32::EPSILON);

        state.set("Speed", Value::Float(0.5));
        response.clear();

        let time = idle_run.evaluate(&mut state, &mut response);
        assert!((time - 3.5).abs() < std::f32::EPSILON);
        assert!((state.get("Speed").as_float() - 0.5).abs() < std::f32::EPSILON);
        assert_eq!(response.len(), 2);
        assert!((response[0].weight - 0.5).abs() < std::f32::EPSILON);
        assert!((response[1].weight - 0.5).abs() < std::f32::EPSILON);

        state.set("Speed", Value::Float(1.0));
        response.clear();

        let time = idle_run.evaluate(&mut state, &mut response);
        assert!((time - 1.0).abs() < std::f32::EPSILON);
        assert!((state.get("Speed").as_float() - 1.0).abs() < std::f32::EPSILON);
        assert_eq!(response.len(), 2);
        assert!((response[0].weight - 0.0).abs() < std::f32::EPSILON);
        assert!((response[1].weight - 1.0).abs() < std::f32::EPSILON);
    }
}
