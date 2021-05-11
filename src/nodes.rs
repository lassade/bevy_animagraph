use crate::{State, Value};

#[derive(Debug)]
pub struct NodeResponse {
    // TODO: pub clip: Handle<Clip>,
    pub weight: f32,
    pub time_scale: f32,
}

pub trait Node {
    fn evaluate(&self, state: &mut State, response: &mut Vec<NodeResponse>) -> f32;
}

pub struct FloatSwitch {
    pub name: String,
    a: Box<dyn Node>,
    b: Box<dyn Node>,
}

impl Node for FloatSwitch {
    fn evaluate(&self, state: &mut State, response: &mut Vec<NodeResponse>) -> f32 {
        let value = state.entry(&self.name, Value::Float(0.0)).as_float();
        float_evaluate(value, &*self.a, &*self.b, state, response)
    }
}

#[inline]
pub(crate) fn float_evaluate(
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
    pub name: String,
    on: Box<dyn Node>,
    off: Box<dyn Node>,
}

impl Node for BoolSwitch {
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
    pub name: String,
    nodes: Vec<(Box<dyn Node>, f32)>,
}

impl Node for Linear1DSwitch {
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
//     pub x: Linear1DSwitch,
//     pub y: Linear1DSwitch,
// }

// impl Node for Linear2DSwitch {
//     fn evaluate(&self, response: &mut Vec<NodeResponse>) {
//         todo!()
//     }
// }

pub struct Animation {
    time: f32,
    // TODO: clip: Handle<Clip>,
}

impl Node for Animation {
    fn evaluate(&self, _: &mut State, response: &mut Vec<NodeResponse>) -> f32 {
        response.push(NodeResponse {
            // TODO: clip: self.clip,
            weight: 1.0,
            time_scale: 1.0,
        });

        self.time
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn float_switch() {
        let idle_run = FloatSwitch {
            name: "Speed".to_string(),
            a: Box::new(Animation { time: 6.0 }),
            b: Box::new(Animation { time: 1.0 }),
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
