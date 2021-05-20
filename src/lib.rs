pub extern crate indexmap;
pub extern crate petgraph;

use std::{f32::EPSILON, iter::Iterator};

use asset_ref::AssetRef;
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
    utils::HashSet,
};
use petgraph::{visit::EdgeRef, EdgeDirection::Outgoing, Graph};
use rand::{thread_rng, Rng};
use serde::{Deserialize, Serialize};
use smallvec::{smallvec, SmallVec};

pub mod asset_ref;
mod id;

type IndexMap<K, V> = indexmap::IndexMap<K, V, ahash::RandomState>;

type IndexSet<K> = indexmap::IndexSet<K, ahash::RandomState>;

///////////////////////////////////////////////////////////////////////////////

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum VarType {
    Bool,
    Float,
}

/// [`AnimatorGraph`] variable, that can be a parameter a fixed value
///
/// **NOTE** It's important to preserve the values to improve the editor experience
#[derive(Default, Debug, Clone, Serialize, Deserialize)]
pub struct Var<T: Default> {
    pub use_param: bool,
    pub value: T,
    pub param: ParamId,
}

impl<T: Default> Var<T> {
    pub fn from_value(value: T) -> Self {
        Self {
            use_param: false,
            value,
            param: ParamId::default(),
        }
    }

    pub fn from_param(param: ParamId) -> Self {
        Self {
            use_param: false,
            value: Default::default(),
            param,
        }
    }
}

pub trait VarTrait<T> {
    fn get(&self, params: &Parameters) -> Option<T>;
    fn var_type() -> VarType;
}

impl VarTrait<f32> for Var<f32> {
    #[inline]
    fn get(&self, params: &Parameters) -> Option<f32> {
        if self.use_param {
            params.get_by_id(self.param).and_then(|(_, p)| p.as_float())
        } else {
            Some(self.value)
        }
    }

    #[inline]
    fn var_type() -> VarType {
        VarType::Float
    }
}

impl VarTrait<bool> for Var<bool> {
    #[inline]
    fn get(&self, params: &Parameters) -> Option<bool> {
        if self.use_param {
            params.get_by_id(self.param).and_then(|(_, p)| p.as_bool())
        } else {
            Some(self.value)
        }
    }

    #[inline]
    fn var_type() -> VarType {
        VarType::Bool
    }
}

///////////////////////////////////////////////////////////////////////////////

/// [`AnimatorGraph`] param, used to feed info to the graph
#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
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

#[derive(Default, Debug, Hash, PartialEq, Eq, Copy, Clone, Serialize, Deserialize)]
pub struct ParamId(Id);

#[derive(Default, Debug, Clone, Serialize, Deserialize)]
pub struct Parameters {
    map: IndexMap<String, Param>,
    ids: Ids,
}

impl Parameters {
    #[inline]
    pub fn get_by_name(&self, name: &str) -> Option<Param> {
        self.map.get(name).copied()
    }

    pub fn find_id(&self, name: &str) -> Option<ParamId> {
        self.map
            .get_full(name)
            .map(|(i, _, _)| ParamId(self.ids.0[i]))
    }

    pub fn get_by_id(&self, param_id: ParamId) -> Option<(&String, &Param)> {
        self.ids
            .find_index(param_id.0)
            .and_then(|i| self.map.get_index(i))
    }

    pub fn remove_by_name(&mut self, name: &str) -> Option<Param> {
        if let Some((index, _, param)) = self.map.swap_remove_full(name) {
            self.ids.0.swap_remove_index(index);
            Some(param)
        } else {
            None
        }
    }

    // pub fn remove_by_index(&mut self, index: usize) -> Option<Param> {
    //     self.map.swap_remove_full(key)
    // }

    // pub fn remove_by_id(&mut self, id: usize) -> Option<Param> {
    //     todo!()
    // }

    pub fn rename_by_name(&mut self, name: &str, target: String) {
        if target.is_empty() {
            return;
        }
        if let Some((a, _, param)) = self.map.swap_remove_full(name) {
            let b = self.map.len();
            self.map.insert(target, param);
            self.map.swap_indices(a, b);
        }
    }

    pub fn insert(&mut self, name: String, param: Param) -> ParamId {
        let (index, _) = self.map.insert_full(name, param);
        if let Some(id) = self.ids.0.get_index(index) {
            ParamId(*id)
        } else {
            ParamId(self.ids.insert())
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn iter(&self) -> impl Iterator<Item = (ParamId, &String, &Param)> {
        let ids_ptr = &self.ids as *const _;
        self.map.iter().enumerate().map(move |(i, (name, param))| {
            // TODO: Borrow won't let me use `self.ids` not sure why
            let ids: &Ids = unsafe { &*ids_ptr };
            (ParamId(ids.0[i]), name, param)
        })
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = (ParamId, &String, &mut Param)> {
        let ids_ptr = &self.ids as *const _;
        self.map
            .iter_mut()
            .enumerate()
            .map(move |(i, (name, param))| {
                // TODO: Borrow won't let me use `self.ids` not sure why
                let ids: &Ids = unsafe { &*ids_ptr };
                (ParamId(ids.0[i]), name, param)
            })
    }
}

///////////////////////////////////////////////////////////////////////////////

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, Serialize, Deserialize)]
struct Id(usize);

impl Default for Id {
    fn default() -> Self {
        Self(0)
    }
}

/// Stable indexes lookup table helper to reuse animator layers
#[derive(Default, Debug, Clone, Serialize, Deserialize)]
struct Ids(IndexSet<Id>);

impl Ids {
    #[inline]
    fn find_index(&self, id: Id) -> Option<usize> {
        self.0.get_full(&id).map(|(i, _)| i)
    }

    fn insert(&mut self) -> Id {
        // Create a new stable index
        let mut rng = thread_rng();
        loop {
            let id = Id(rng.gen::<usize>().max(1));
            if self.0.insert(id) {
                return id;
            }
        }
    }
}

///////////////////////////////////////////////////////////////////////////////

#[derive(Debug, TypeUuid, Serialize, Deserialize)]
#[uuid = "6b7c940d-a698-40ae-9ff2-b08747d6e8e1"]
pub struct AnimaGraph {
    pub name: String,
    pub parameters: Parameters,
    pub layers: Vec<Layer>,
}

impl Default for AnimaGraph {
    fn default() -> Self {
        AnimaGraph {
            name: String::default(),
            parameters: Parameters::default(),
            layers: vec![Layer {
                name: "Layer0".to_string(),
                ..Default::default()
            }],
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Layer {
    pub name: String,
    pub default_weight: f32,
    pub additive: bool,
    pub graph: Graph<State, Transition>,
}

impl Default for Layer {
    fn default() -> Self {
        Layer {
            name: String::default(),
            default_weight: 1.0,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct State {
    pub name: String,
    /// State position, used for rendering the graph
    pub position: Vec2,
    /// Normalized state start time
    pub offset: Var<f32>,
    pub time_scale: Var<f32>,
    /// State data
    pub data: StateData,
}

impl Default for State {
    fn default() -> Self {
        State {
            name: String::default(),
            position: Vec2::ZERO,
            offset: Var::from_value(0.0),
            time_scale: Var::from_value(1.0),
            data: StateData::Marker,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StateData {
    /// Used for `Entry`, `Exit` or `Any`
    Marker,
    Clip {
        clip: AssetRef<Clip>,
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

#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub enum Distance {
    Block,
    Cartesian,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entry<T> {
    pub clip: AssetRef<Clip>,
    pub position: T,
    pub time_scale: Var<f32>,
}

#[derive(Default, Debug, Clone, Serialize, Deserialize)]
pub struct Blend1D {
    entries: Vec<Entry<f32>>,
}

#[derive(Default, Debug, Clone, Serialize, Deserialize)]
pub struct Blend2D {
    entries: Vec<Entry<Vec2>>,
}

#[derive(Debug, PartialEq, Eq, Copy, Clone, Serialize, Deserialize)]
pub enum Compare {
    Equal,
    Less,
    LessOrEqual,
    Greater,
    GreaterOrEqual,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Condition {
    Bool {
        x: ParamId,
        y: Var<bool>,
    },
    Float {
        x: ParamId,
        op: Compare,
        y: Var<f32>,
    },
    ExitTime {
        time: f32,
    },
    ExitTimeNormalized {
        normalized_time: f32,
    },
}

#[derive(Default, Debug, Clone, Serialize, Deserialize)]
pub struct Transition {
    pub length: f32,
    pub conditions: Vec<Condition>,
}

///////////////////////////////////////////////////////////////////////////////

#[derive(Default, Debug, Clone)]
pub struct StateInfo {
    state: u32,
    duration: f32,
    owned: SmallVec<[Id; 6]>,
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
    params: Parameters,
    ids: Ids,
}

impl GraphInfo {
    pub fn layers(&self) -> &[LayerInfo] {
        &self.layers[..]
    }

    pub fn params(&self) -> &Parameters {
        &self.params
    }

    pub fn set_param_by_name<S>(&mut self, name: &str, param: Param) {
        match self.params.map.get_mut(name) {
            Some(Param::Float(v)) => *v = param.as_float().unwrap_or(0.0),
            Some(Param::Bool(v)) => *v = param.as_bool().unwrap_or(false),
            _ => {}
        }
    }

    pub fn set_param_by_id<S>(&mut self, id: ParamId, param: Param) {
        if let Some(index) = self.params.ids.find_index(id.0) {
            match self.params.map.get_index_mut(index) {
                Some((_, Param::Float(v))) => *v = param.as_float().unwrap_or(0.0),
                Some((_, Param::Bool(v))) => *v = param.as_bool().unwrap_or(false),
                _ => {}
            }
        }
    }
}

#[derive(Debug, Reflect)]
#[reflect(Component)]
pub struct AnimaGraphPlayer {
    graph: Handle<AnimaGraph>,
    #[reflect(ignore)]
    runtime: Option<GraphInfo>,
    pub time_scale: f32,
}

impl Default for AnimaGraphPlayer {
    fn default() -> Self {
        AnimaGraphPlayer {
            graph: Handle::default(),
            runtime: None,
            time_scale: 1.0,
        }
    }
}

impl AnimaGraphPlayer {
    pub fn set_graph(&mut self, graph: Handle<AnimaGraph>) {
        self.graph = graph;
        self.runtime = None;
    }

    pub fn graph(&self) -> &Handle<AnimaGraph> {
        &self.graph
    }

    pub fn runtime(&self) -> &Option<GraphInfo> {
        &self.runtime
    }
}

pub(crate) fn animator_controller_system(
    time: Res<Time>,
    clips: Res<Assets<Clip>>,
    graphs: Res<Assets<AnimaGraph>>,
    //mut clip_events: EventReader<AssetEvent<Clip>>,
    mut graph_events: EventReader<AssetEvent<AnimaGraph>>,
    mut controllers: Query<(&mut Animator, &mut AnimaGraphPlayer)>,
) {
    let clips = &*clips;
    let delta_time = time.delta_seconds();

    // Query all graphs that need to be invalidated
    let mut invalidate: HashSet<Handle<AnimaGraph>> = HashSet::default();
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

        let controller: &mut AnimaGraphPlayer = &mut *controller;
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
                let params = graph.parameters.clone();

                // Create layers
                let layers = graph
                    .layers
                    .iter()
                    .map(|layer| LayerInfo {
                        current_state: StateInfo::default(),
                        transition: None,
                        weight: layer.default_weight,
                    })
                    .collect();

                GraphInfo {
                    layers,
                    params,
                    ids: Ids::default(),
                }
            });

            // Needed for the borrow checker
            let layers = &mut runtime.layers;
            let parameters = &mut runtime.params;
            let ids = &mut runtime.ids;
            let mut stack_index = 0;

            // Execute
            for (layer_index, layer_info) in layers.iter_mut().enumerate() {
                let layer = &graph.layers[layer_index];

                // Process transitions
                update_transition(animator, ids, parameters, layer, layer_info, delta_time);

                // Process states
                let layer_weight = layer_info.weight;
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
                            ids,
                            &mut stack_index,
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
                        ids,
                        &mut stack_index,
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
    animator: &mut Animator,
    ids: &mut Ids,
    parameters: &Parameters,
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

            // TODO: Other interrupt sources
            if transition_info.normalized_time > 1.0 {
                // Transition is done

                // Remove animator layers
                let current_state = &mut layer_info.current_state;
                for i in (0..current_state.owned.len()).rev() {
                    remove_animator_layer(animator, ids, &mut current_state.owned, i);
                }

                // Copy state info
                *current_state = transition_info.to.clone();

                // Clear transition
                layer_info.transition = None;
            } else {
                // Transition still in progress
                return;
            }
        }
    }

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
                        let x = parameters.get_by_id(*x).and_then(|(_, p)| p.as_bool());
                        let y = y.get(parameters);
                        x == y
                    }
                    Condition::Float { x, op, y } => {
                        let x = parameters
                            .get_by_id(*x)
                            .and_then(|(_, p)| p.as_float())
                            .unwrap_or(0.0);
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
                    owned: smallvec![],
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

fn push_animator_layer<'a>(
    animator: &'a mut Animator,
    ids: &mut Ids,
    stack_index: &mut usize,
    owned: &mut SmallVec<[Id; 6]>,
    owned_index: &mut usize,
) -> &'a mut bevy::animation::Layer {
    // Find where the layer is at
    let b = if let Some(id) = owned.get(*owned_index) {
        ids.find_index(*id).unwrap()
    } else {
        // Layer not found, allocate a new one
        let id = ids.insert();
        owned.push(id);

        let index = animator.layers.len();
        let layer = bevy::animation::Layer::default();
        animator.layers.push(layer);
        index
    };
    *owned_index += 1;

    // Put layer in the right order
    let a = *stack_index;
    animator.layers.swap(a, b);
    ids.0.swap_indices(a, b);
    *stack_index += 1;

    &mut animator.layers[a]
}

fn remove_animator_layer<'a>(
    animator: &'a mut Animator,
    ids: &mut Ids,
    owned: &mut SmallVec<[Id; 6]>,
    owned_index: usize,
) {
    // Find where the layer is at
    let id = owned.remove(owned_index);
    let (index, _) = ids.0.swap_remove_full(&id).unwrap();
    animator.layers.swap_remove(index);
}

fn update_state(
    animator: &mut Animator,
    ids: &mut Ids,
    stack_index: &mut usize,
    clips: &Assets<Clip>,
    parameters: &Parameters,
    state: &State,
    state_info: &mut StateInfo,
    delta_time: f32,
    weight: f32,
    additive: bool,
) {
    let time_scale = state.time_scale.get(parameters).unwrap_or(1.0);
    let mut owned_index = 0;
    match &state.data {
        StateData::Marker => {}
        StateData::Clip { clip } => {
            let clip_handle = clip;
            if let Some(clip) = clips.get(clip.as_ref()) {
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
                let clip = animator.add_clip(clip_handle.clone().into()); // TODO: add_clip should take a ref to handle;
                let layer = push_animator_layer(
                    animator,
                    ids,
                    stack_index,
                    &mut state_info.owned,
                    &mut owned_index,
                );
                layer.weight = weight;
                layer.clip = clip;
                layer.time = n * d;
                layer.time_scale = 0.0;
                layer.additive = additive;
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
        app.add_asset::<AnimaGraph>()
            .register_type::<AnimaGraphPlayer>()
            .add_system_to_stage(
                AnimationStage::Animate,
                animator_controller_system
                    .system()
                    .before(AnimationSystem::Animate),
            );
    }
}
