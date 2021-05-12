use bevy::{
    animation::Clip,
    app::EventReader,
    asset::{AssetEvent, Assets, Handle},
    ecs::system::{Query, Res},
    math::Vec2,
    reflect::TypeUuid,
    utils::HashMap,
};
use petgraph::Graph;

///////////////////////////////////////////////////////////////////////////////

/// [`AnimatorGraph`] variable, that can be a parameter a fixed value
#[derive(Debug)]
pub enum Var<T> {
    Value(T),
    Param(String),
}

impl Var<f32> {
    #[inline]
    pub fn get(&self, params: &HashMap<String, Param>) -> Option<f32> {
        match self {
            Var::Value(value) => Some(*value),
            Var::Param(name) => params.get(name).map(|param| param.as_float()).flatten(),
        }
    }

    #[inline]
    pub fn get_or_insert_default(&self, params: &mut HashMap<String, Param>) -> Option<f32> {
        match self {
            Var::Value(value) => Some(*value),
            Var::Param(name) => {
                if let Some(param) = params.get(name) {
                    param.as_float()
                } else {
                    params.insert(name.clone(), Param::Float(0.0));
                    Some(0.0)
                }
            }
        }
    }
}

impl Var<bool> {
    #[inline]
    pub fn get(&self, params: &HashMap<String, Param>) -> Option<bool> {
        match self {
            Var::Value(value) => Some(*value),
            Var::Param(name) => params.get(name).map(|param| param.as_bool()).flatten(),
        }
    }

    #[inline]
    pub fn get_or_insert_default(&self, params: &mut HashMap<String, Param>) -> Option<bool> {
        match self {
            Var::Value(value) => Some(*value),
            Var::Param(name) => {
                if let Some(param) = params.get(name) {
                    param.as_bool()
                } else {
                    params.insert(name.clone(), Param::Bool(false));
                    Some(false)
                }
            }
        }
    }
}

///////////////////////////////////////////////////////////////////////////////

/// [`AnimatorGraph`] param, used to feed info to the graph
#[derive(Debug, Copy, Clone)]
pub enum Param {
    Bool(bool),
    Float(f32),
}

impl Param {
    #[inline]
    pub fn as_float(&self) -> Option<f32> {
        if let Param::Float(value) = self {
            Some(*value)
        } else {
            None
        }
    }

    #[inline]
    pub fn as_bool(&self) -> Option<bool> {
        if let Param::Bool(value) = self {
            Some(*value)
        } else {
            None
        }
    }
}

///////////////////////////////////////////////////////////////////////////////

#[derive(Debug, TypeUuid)]
#[uuid = "6b7c940d-a698-40ae-9ff2-b08747d6e8e1"]
pub struct AnimatorGraph {
    parameters: HashMap<String, Param>,
    layers: Vec<Layer>,
}

#[derive(Debug)]
pub struct Layer {
    pub name: String,
    pub weight: Var<f32>,
    graph: Graph<State, Transition>,
}

#[derive(Debug)]
pub struct State {
    pub name: String,
    pub data: StateData,
}

#[derive(Debug)]
pub enum StateData {
    /// Used for `Entry`, `Exit` or `Any`
    Marker,
    Clip {
        offset: Var<f32>,
        clip: Handle<Clip>,
        time_scale: Var<f32>,
    },
    // SubGraph {
    //     graph: Graph<State, Transition>,
    // },
    Blend1D {
        offset: Var<f32>,
        value: Var<f32>,
        entries: Vec<Entry<f32>>,
    },
    Blend2D {
        mode: Distance,
        offset: Var<f32>,
        x: Var<f32>,
        y: Var<f32>,
        entries: Vec<Entry<Vec2>>,
    },
}

#[derive(Debug)]
pub enum Distance {
    Block,
    Cartesian,
}

#[derive(Debug)]
pub struct Entry<T> {
    pub clip: Handle<Clip>,
    pub position: T,
    pub time_scale: Var<f32>,
}

#[derive(Debug)]
pub enum Compare {
    Equal,
    Less,
    LessOrEqual,
    Greater,
    GreaterOrEqual,
}

#[derive(Debug)]
pub enum Condition {
    Bool { x: String, y: Var<bool> },
    Float { x: String, op: Compare, y: Var<f32> },
}

#[derive(Debug)]
pub struct Transition {
    pub length: f32,
    pub conditions: Vec<Condition>,
}

///////////////////////////////////////////////////////////////////////////////

#[derive(Default, Debug)]
pub struct StateInfo {
    state: u32,
    duration: f32,
    pub normalized_time: f32,
}

#[derive(Debug)]
pub struct TransitionInfo {
    index: u32,
    to: StateInfo,
    pub time: f32,
}

#[derive(Debug)]
pub struct LayerInfo {
    current_state: StateInfo,
    transition: Option<TransitionInfo>,
    pub weight: f32,
}

#[derive(Debug)]
pub struct GraphInfo {
    layers: Vec<LayerInfo>,
    parameters: HashMap<String, Param>,
}

#[derive(Debug)]
pub struct AnimatorController {
    graph: Handle<AnimatorGraph>,
    runtime: Option<GraphInfo>,
}

impl AnimatorController {
    pub fn set_graph(&mut self, graph: Handle<AnimatorGraph>) {
        self.graph = graph;
        self.runtime = None;
    }

    pub fn graph(&self) -> &Handle<AnimatorGraph> {
        &self.graph
    }
}

pub(crate) fn animator_controller_system(
    clips: Res<Assets<Clip>>,
    graphs: Res<Assets<AnimatorGraph>>,
    //mut clip_events: EventReader<AssetEvent<Clip>>,
    mut graph_events: EventReader<AssetEvent<AnimatorGraph>>,
    mut controllers: Query<&mut AnimatorController>,
) {
    for mut controller in controllers.iter_mut() {
        let controller: &mut AnimatorController = &mut *controller;

        if let Some(graph) = graphs.get(&controller.graph) {
            // Get or create the runtime
            let runtime = controller.runtime.get_or_insert_with(|| {
                // Default parameters
                let mut parameters = graph.parameters.clone();

                // Create layers
                let layers = graph
                    .layers
                    .iter()
                    .map(|layer| LayerInfo {
                        current_state: StateInfo::default(),
                        transition: None,
                        weight: layer
                            .weight
                            .get_or_insert_default(&mut parameters)
                            .unwrap_or(0.0),
                    })
                    .collect();

                GraphInfo { layers, parameters }
            });

            // Execute
        }
    }
}
