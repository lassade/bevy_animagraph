pub extern crate petgraph;

use std::f32::EPSILON;

use bevy::{
    animation::{AnimationStage, AnimationSystem, Animator, Clip},
    app::{EventReader, Plugin},
    asset::{AddAsset, AssetEvent, Assets, Handle},
    core::Time,
    ecs::{
        reflect::ReflectComponent,
        schedule::ParallelSystemDescriptorCoercion,
        system::{IntoSystem, Query, Res},
    },
    math::Vec2,
    reflect::{Reflect, TypeUuid},
    utils::{HashMap, HashSet},
};
use petgraph::{visit::EdgeRef, EdgeDirection::Outgoing, Graph};

///////////////////////////////////////////////////////////////////////////////

/// [`AnimatorGraph`] variable, that can be a parameter a fixed value
#[derive(Debug, Clone)]
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
pub struct Animagraph {
    pub name: String,
    pub parameters: HashMap<String, Param>,
    pub layers: Vec<Layer>,
}

impl Default for Animagraph {
    fn default() -> Self {
        Animagraph {
            name: String::default(),
            parameters: HashMap::default(),
            layers: vec![Layer {
                name: "Layer0".to_string(),
                ..Default::default()
            }],
        }
    }
}

#[derive(Debug)]
pub struct Layer {
    pub name: String,
    pub weight: Var<f32>,
    pub additive: bool,
    pub graph: Graph<State, Transition>,
}

impl Default for Layer {
    fn default() -> Self {
        Layer {
            name: String::default(),
            weight: Var::Value(1.0),
            additive: false,
            graph: Graph::new(),
        }
    }
}

impl Layer {
    pub fn set_default_state(&mut self, state: u32) -> bool {
        if state == 0 || (state as usize) >= self.graph.node_count() {
            return false;
        }

        // TODO: Theres got to be a better way of doing this

        let mut temp = Graph::with_capacity(self.graph.node_count(), self.graph.edge_count());
        std::mem::swap(&mut temp, &mut self.graph);

        let (mut nodes, edges) = temp.into_nodes_edges();
        nodes[..].swap(0, state as usize);

        let default_state = 0.into();
        let state = state.into();

        for node in nodes {
            self.graph.add_node(node.weight);
        }

        for edge in edges {
            let mut source = edge.source();
            if source == state {
                source = default_state;
            } else if source == default_state {
                source = state;
            }

            let mut target = edge.target();
            if target == state {
                target = default_state;
            } else if target == default_state {
                target = state;
            }

            self.graph.add_edge(source, target, edge.weight);
        }

        true
    }
}

#[derive(Debug, Clone)]
pub struct State {
    pub name: String,
    /// State position, used for rendering the graph
    pub position: Vec2,
    /// Normalized state start time
    pub offset: Var<f32>,
    /// State data
    pub data: StateData,
}

impl Default for State {
    fn default() -> Self {
        State {
            name: String::default(),
            position: Vec2::ZERO,
            offset: Var::Value(0.0),
            data: StateData::Marker,
        }
    }
}

#[derive(Debug, Clone)]
pub enum StateData {
    /// Used for `Entry`, `Exit` or `Any`
    Marker,
    Clip {
        clip: Handle<Clip>,
        time_scale: Var<f32>,
    },
    // SubGraph {
    //     graph: Graph<State, Transition>,
    // },
    Blend1D {
        value: Var<f32>,
        blend: Blend1D,
    },
    Blend2D {
        mode: Distance,
        x: Var<f32>,
        y: Var<f32>,
        blend: Blend2D,
    },
}

#[derive(Debug, Copy, Clone)]
pub enum Distance {
    Block,
    Cartesian,
}

#[derive(Debug, Clone)]
pub struct Entry<T> {
    pub clip: Handle<Clip>,
    pub position: T,
    pub time_scale: Var<f32>,
}

#[derive(Default, Debug, Clone)]
pub struct Blend1D {
    entries: Vec<Entry<f32>>,
}

#[derive(Default, Debug, Clone)]
pub struct Blend2D {
    entries: Vec<Entry<Vec2>>,
}

#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub enum Compare {
    Equal,
    Less,
    LessOrEqual,
    Greater,
    GreaterOrEqual,
}

#[derive(Debug, Clone)]
pub enum Condition {
    Bool { x: String, y: Var<bool> },
    Float { x: String, op: Compare, y: Var<f32> },
    ExitTime { time: f32 },
    ExitTimeNormalized { normalized_time: f32 },
}

#[derive(Default, Debug, Clone)]
pub struct Transition {
    pub length: f32,
    pub conditions: Vec<Condition>,
}

///////////////////////////////////////////////////////////////////////////////

#[derive(Default, Debug, Copy, Clone)]
pub struct StateInfo {
    state: u32,
    duration: f32,
    pub loop_count: u32,
    pub normalized_time: f32,
}

impl StateInfo {
    pub fn state(&self) -> u32 {
        self.state
    }

    pub fn duration(&self) -> f32 {
        self.duration
    }
}

#[derive(Debug)]
pub struct TransitionInfo {
    transition: u32,
    to: StateInfo,
    pub normalized_time: f32,
}

impl TransitionInfo {
    pub fn transition(&self) -> u32 {
        self.transition
    }

    pub fn to(&self) -> &StateInfo {
        &self.to
    }
}

#[derive(Debug)]
pub struct LayerInfo {
    current_state: StateInfo,
    transition: Option<TransitionInfo>,
    pub weight: f32,
}

impl LayerInfo {
    pub fn current_state(&self) -> &StateInfo {
        &self.current_state
    }

    pub fn transition(&self) -> Option<&TransitionInfo> {
        self.transition.as_ref()
    }
}

#[derive(Debug)]
pub struct GraphInfo {
    layers: Vec<LayerInfo>,
    params: HashMap<String, Param>,
}

impl GraphInfo {
    pub fn layers(&self) -> &[LayerInfo] {
        &self.layers[..]
    }

    pub fn params(&self) -> &HashMap<String, Param> {
        &self.params
    }

    pub fn set_param<S>(&mut self, name: &str, param: Param) {
        match self.params.get_mut(name) {
            Some(Param::Float(v)) => *v = param.as_float().unwrap_or(0.0),
            Some(Param::Bool(v)) => *v = param.as_bool().unwrap_or(false),
            _ => {}
        }
    }
}

#[derive(Debug, Reflect)]
#[reflect(Component)]
pub struct AnimagraphPlayer {
    graph: Handle<Animagraph>,
    #[reflect(ignore)]
    runtime: Option<GraphInfo>,
    pub time_scale: f32,
}

impl Default for AnimagraphPlayer {
    fn default() -> Self {
        AnimagraphPlayer {
            graph: Handle::default(),
            runtime: None,
            time_scale: 1.0,
        }
    }
}

impl AnimagraphPlayer {
    pub fn set_graph(&mut self, graph: Handle<Animagraph>) {
        self.graph = graph;
        self.runtime = None;
    }

    pub fn graph(&self) -> &Handle<Animagraph> {
        &self.graph
    }

    pub fn runtime(&self) -> &Option<GraphInfo> {
        &self.runtime
    }
}

pub(crate) fn animator_controller_system(
    time: Res<Time>,
    clips: Res<Assets<Clip>>,
    graphs: Res<Assets<Animagraph>>,
    //mut clip_events: EventReader<AssetEvent<Clip>>,
    mut graph_events: EventReader<AssetEvent<Animagraph>>,
    mut controllers: Query<(&mut Animator, &mut AnimagraphPlayer)>,
) {
    let clips = &*clips;
    let delta_time = time.delta_seconds();

    // Query all graphs that need to be invalidated
    let mut invalidate: HashSet<Handle<Animagraph>> = HashSet::default();
    for event in graph_events.iter() {
        match event {
            AssetEvent::Created { handle }
            | AssetEvent::Modified { handle }
            | AssetEvent::Removed { handle } => {
                invalidate.insert(handle.clone());
            }
        }
    }

    for (mut animator, mut controller) in controllers.iter_mut() {
        let animator: &mut Animator = &mut *animator;

        let controller: &mut AnimagraphPlayer = &mut *controller;
        let delta_time = delta_time * controller.time_scale;

        // Invalidate controller because it was changed,
        // ? NOTE: sadly this is the least worst thing that can happen,
        // ? otherwise it will be left to chance that the runtime animator
        // ? enters in a invalid state configuration, one it can't leave or recover from;
        if invalidate.contains(&controller.graph) {
            controller.runtime = None;
        }

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

                GraphInfo {
                    layers,
                    params: parameters,
                }
            });

            // Clear previous frame layers
            animator.layers.clear();

            // Needed for the borrow checker
            let layers = &mut runtime.layers;
            let parameters = &mut runtime.params;

            // Execute
            for (layer_index, layer_info) in layers.iter_mut().enumerate() {
                let layer = &graph.layers[layer_index];

                // Process transitions
                update_transition(parameters, layer, layer_info, delta_time);

                // Process states
                let layer_weight = layer.weight.get(parameters).unwrap_or(1.0);
                let mut weight = 1.0;

                if let Some(transition_info) = &layer_info.transition {
                    weight = transition_info.normalized_time;

                    if let Some(state_node) = layer
                        .graph
                        .raw_nodes()
                        .get(transition_info.to.state as usize)
                    {
                        update_state(
                            animator,
                            clips,
                            parameters,
                            &state_node.weight,
                            &mut layer_info.current_state,
                            delta_time,
                            weight * layer_weight,
                            layer.additive,
                        );
                    }

                    weight = 1.0 - weight;
                }

                if let Some(state_node) = layer
                    .graph
                    .raw_nodes()
                    .get(layer_info.current_state.state as usize)
                {
                    update_state(
                        animator,
                        clips,
                        parameters,
                        &state_node.weight,
                        &mut layer_info.current_state,
                        delta_time,
                        weight * layer_weight,
                        layer.additive,
                    );
                }
            }
        }
    }
}

fn update_transition(
    parameters: &HashMap<String, Param>,
    layer: &Layer,
    layer_info: &mut LayerInfo,
    delta_time: f32,
) {
    if let Some(transition_info) = &mut layer_info.transition {
        if let Some(transition_edge) = layer
            .graph
            .raw_edges()
            .get(transition_info.transition as usize)
        {
            transition_info.normalized_time += delta_time / transition_edge.weight.length;

            if transition_info.normalized_time > 1.0 {
                // Transition is done
                layer_info.current_state = transition_info.to;
            } else {
                // Transition still in progress
                // TODO: Interrupt sources
                return;
            }
        }
    }

    // Clear transition
    layer_info.transition = None;

    // Check for transitions in the current state
    if let Some(transition_edge) = layer
        .graph
        .edges_directed(layer_info.current_state.state.into(), Outgoing)
        .find(|transition_edge| {
            // Check if transition met all their conditions
            transition_edge
                .weight()
                .conditions
                .iter()
                .all(|condition| match condition {
                    Condition::Bool { x, y } => {
                        let x = parameters.get(x).and_then(Param::as_bool);
                        let y = y.get(parameters);
                        x == y
                    }
                    Condition::Float { x, op, y } => {
                        let x = parameters.get(x).and_then(Param::as_float).unwrap_or(0.0);
                        let y = y.get(parameters).unwrap_or(0.0);
                        match op {
                            Compare::Equal => (x - y).abs() < EPSILON,
                            Compare::Less => x < y,
                            Compare::LessOrEqual => x < (y + EPSILON),
                            Compare::Greater => x > y,
                            Compare::GreaterOrEqual => x > (y - EPSILON),
                        }
                    }
                    Condition::ExitTime { time } => {
                        let state_info = &layer_info.current_state;
                        let normalized_total_time =
                            state_info.normalized_time + state_info.loop_count as f32;
                        *time > normalized_total_time * state_info.duration
                    }
                    Condition::ExitTimeNormalized { normalized_time } => {
                        let state_info = &layer_info.current_state;
                        *normalized_time
                            > (state_info.normalized_time + state_info.loop_count as f32)
                    }
                })
        })
    {
        // Start a new transition
        let state_index = transition_edge.target().index();
        if let Some(state_node) = layer.graph.raw_nodes().get(state_index) {
            layer_info.transition = Some(TransitionInfo {
                transition: transition_edge.id().index() as u32,
                to: StateInfo {
                    state: state_index as u32,
                    duration: 0.0,
                    loop_count: 0,
                    normalized_time: state_node
                        .weight
                        .offset
                        .get(parameters)
                        .unwrap_or(0.0)
                        .fract(),
                },
                normalized_time: 0.0,
            });
        } else {
            // TODO: Invalid transition
        }
    }
}

fn update_state(
    animator: &mut Animator,
    clips: &Assets<Clip>,
    parameters: &HashMap<String, Param>,
    state: &State,
    state_info: &mut StateInfo,
    delta_time: f32,
    weight: f32,
    additive: bool,
) {
    match &state.data {
        StateData::Marker => {}
        StateData::Clip { clip, time_scale } => {
            let clip_handle = clip;
            if let Some(clip) = clips.get(clip) {
                let time_scale = time_scale.get(parameters).unwrap_or(1.0);
                let d = clip.duration();

                let mut n = state_info.normalized_time;
                n += (delta_time * time_scale) / d;
                if n > 1.0 {
                    if clip.warp {
                        n = n.fract();
                        state_info.loop_count += 1;
                    } else {
                        n = 1.0;
                    }
                }

                state_info.duration = d;
                state_info.normalized_time = n;

                // Add layer
                let mut layer = bevy::animation::Layer::default();
                layer.weight = weight;
                layer.clip = animator.add_clip(clip_handle.clone()); // TODO: add_clip should take a ref to handle
                layer.time = n * d;
                layer.time_scale = 0.0;
                layer.additive = additive;
                animator.layers.push(layer);
            }
        }
        StateData::Blend1D {
            value,
            blend: entries,
        } => {
            let _ = value;
            let _ = entries;
            todo!()
        }
        StateData::Blend2D {
            mode,
            x,
            y,
            blend: entries,
        } => {
            let _ = mode;
            let _ = x;
            let _ = y;
            let _ = entries;
            todo!()
        }
    }
}

pub struct AnimatorControllerPlugin;

impl Plugin for AnimatorControllerPlugin {
    fn build(&self, app: &mut bevy::prelude::AppBuilder) {
        app.add_asset::<Animagraph>()
            .register_type::<AnimagraphPlayer>()
            .add_system_to_stage(
                AnimationStage::Animate,
                animator_controller_system
                    .system()
                    .before(AnimationSystem::Animate),
            );
    }
}
