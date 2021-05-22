use std::{f32::EPSILON, fs::File, io::Read, path::Path, string::ToString};

use anyhow::Result;
use bevy::{animation::Clip, asset::{Asset, HandleId}, prelude::*, utils::HashMap};
use bevy_animagraph::{AnimaGraph, Layer, Param, ParamId, Parameters, State, StateData, Transition, Var, VarType, asset_ref::{AssetRef, AssetSerializer}, petgraph::{visit::EdgeRef, EdgeDirection::Outgoing}};
use bevy_egui::{
    egui::{
        self, pos2, vec2, Color32, DragValue, Frame, Id, Key, Label, Layout, Pos2, Rect, Response,
        Sense, Stroke, TextEdit, TextStyle, Ui, Vec2, WidgetInfo, WidgetType,
    },
    EguiContext,
};

///////////////////////////////////////////////////////////////////////////////

#[derive(Debug, PartialEq, Eq, Copy, Clone)]
enum Selected {
    State(u32),
    Transition(u32),
}

#[derive(Debug, Clone)]
enum EditOp {
    None,
    AddTransition { state: u32, position: Pos2 },
}

struct TransitionGroup {
    index: u32,
    count: usize,
    p0: Pos2,
    p1: Pos2,
}

#[derive(Default)]
struct Cache {
    target: Option<Handle<AnimaGraph>>,
    target_layer: usize,
    grouped_transitions: HashMap<(u32, u32), TransitionGroup>,
}

impl Cache {
    fn clear(&mut self) {
        self.target = None;
        self.grouped_transitions.clear();
    }
}

#[derive(Default)]
struct Filters {
    clips: String,
}

pub struct AnimaGraphEditor {
    pub open: bool,
    pub editing: Option<Handle<AnimaGraph>>,
    pub live: Option<Entity>,
    operation: EditOp,
    position: Vec2,
    selected: Option<Selected>,
    selected_layer: usize,
    context_menu_position: Pos2,
    temp_buffer: String,
    cache: Cache,
    filters: Filters,
}

const STATE_SIZE: Vec2 = vec2(180.0, 40.0);
const ARROW_SIZE: f32 = 8.0;
const EXTENSIONS: &'static [&'static str] = &["anima_graph"];

fn animator_graph_editor_system(
    mut graph_editor: ResMut<AnimaGraphEditor>,
    mut graphs: ResMut<Assets<AnimaGraph>>,
    clips: ResMut<Assets<Clip>>,
    asset_server: Res<AssetServer>,
    egui_context: Res<EguiContext>,
) {
    let AnimaGraphEditor {
        open,
        editing,
        live,
        operation,
        position,
        selected,
        selected_layer,
        context_menu_position,
        temp_buffer,
        cache,
        filters,
    } = &mut *graph_editor;

    // Clear cache
    if cache.target != *editing || cache.target_layer != *selected_layer {
        cache.clear();
        cache.target = editing.clone();
    }

    // Store graphs pointer for later
    let graphs_ptr = &*graphs as *const _ as *mut Assets<AnimaGraph>;

    enum MenuOp {
        None,
        CreateNew,
        Load,
    }
    let mut menu_op = MenuOp::None;

    egui::Window::new("AnimaGraph Editor")
        .default_size([1100.0, 600.0])
        .open(open)
        .show(egui_context.ctx(), |ui| {
            //let rect = ui.available_rect_before_wrap();
            let (id, mut rect) = ui.allocate_space(ui.available_size());

            if let Some(graph_handle) = editing {
                let graph_id = graph_handle.id;
                if let Some(mut target) = Modify::new(&mut *graphs, &*graph_handle) {
                    let layer_count = target.view().layers.len();
                    if *selected_layer > layer_count {
                        *selected_layer = 0;
                    }

                    if layer_count == 0 {
                        // Create the default layer
                        if action_needed(ui, rect, "Graph have no layers", "Create Layer").clicked()
                        {
                            target.mutate().layers.push(Layer {
                                name: "Layer0".to_string(),
                                ..Default::default()
                            });
                        }
                    } else {
                        let mut defer_modify = false;

                        // Tool bar
                        const TOOLBAR_HEIGHT: f32 = 25.0;
                        let mut toolbar_rect = rect;
                        toolbar_rect.max.y = toolbar_rect.min.y + TOOLBAR_HEIGHT;
                        rect.min.y += TOOLBAR_HEIGHT;
                        ui.allocate_ui_at_rect(toolbar_rect, |ui: &mut Ui| {
                            ui.horizontal(|ui| {
                                egui::ComboBox::from_id_source("graph_menu")
                                    .selected_text("☰ Menu")
                                    .show_ui(ui, |ui| {
                                        if ui.selectable_label(false, "Save As").clicked() {
                                            let save_path = native_dialog::FileDialog::new()
                                                .set_filename(&format!(
                                                    "{}.anima_graph",
                                                    target.view().name
                                                ))
                                                .add_filter("AnimaGraph", EXTENSIONS)
                                                .show_save_single_file();

                                            if let Ok(Some(save_path)) = save_path {
                                                if let Err(err) = save_file(
                                                    save_path.as_path(),
                                                    &asset_server,
                                                    target.view(),
                                                ) {
                                                    error!("`AssetGraph` couldn't be saved at '{}', reason: {}", save_path.display(), err);
                                                }
                                            }
                                        }
                                        if ui.selectable_label(false, "Load").clicked() {
                                            menu_op = MenuOp::Load;
                                        }
                                    });

                                ui.label("Graph");
                                egui::ComboBox::from_id_source("graph_select")
                                    .selected_text(&target.view().name)
                                    .show_ui(ui, |ui| {
                                        // SAFETY: We only want to display all the graphs available for editing
                                        let graphs = unsafe { &*graphs_ptr };
                                        for (other_id, other) in graphs.iter() {
                                            if ui
                                                .selectable_label(other_id == graph_id, &other.name)
                                                .clicked()
                                            {
                                                *editing = Some(Handle::weak(other_id));
                                            }
                                        }
                                        // ? NOTE: Graphs with no strong references will be disposed in the next few frames
                                        if ui.selectable_label(false, "*Create New").clicked() {
                                            menu_op = MenuOp::CreateNew;
                                        }
                                    });

                                // Select active layer
                                ui.label("Layer");
                                egui::ComboBox::from_id_source("layer_select")
                                    .selected_text(&target.view().layers[*selected_layer].name)
                                    .show_ui(ui, |ui| {
                                        for i in 0..layer_count {
                                            let text = &target.view().layers[i].name;
                                            if ui
                                                .selectable_value(selected_layer, i, text)
                                                .clicked()
                                            {
                                                *selected = None;
                                            }
                                        }
                                        if ui.selectable_label(false, "*Create New").clicked() {
                                            let layers = &mut target.mutate().layers;
                                            let layers_count = layers.len();
                                            layers.push(Layer {
                                                name: format!("Layer{}", layers_count),
                                                ..Default::default()
                                            });
                                            *selected_layer = layer_count;
                                        }
                                    });
                            });
                        });

                        // Inspector
                        const INSPECTOR_WIDTH: f32 = 300.0;
                        let mut inspector_rect = rect;
                        // inspector_rect.max.x = inspector_rect.min.x + INSPECTOR_WIDTH - 6.0;
                        // rect.min.x += INSPECTOR_WIDTH;
                        inspector_rect.min.x = inspector_rect.max.x - INSPECTOR_WIDTH;
                        rect.max.x -= INSPECTOR_WIDTH + 6.0;

                        let state_id = id.with("state");
                        ui.allocate_ui_at_rect(inspector_rect, |ui: &mut Ui| match selected {
                            Some(Selected::State(state)) => {
                                let graph = target.mutate_without_modify();
                                let layers = &mut graph.layers;
                                let parameters = &graph.parameters;
                                if let Some(state) = layers[*selected_layer]
                                    .graph
                                    .node_weight_mut((*state).into())
                                {
                                    heading(ui, "State");
                                    field(ui, "Name", |ui| {
                                        ui.text_edit_singleline(&mut state.name);
                                    });
                                    field(ui, "Offset", |ui| {
                                        var_widget(
                                            ui,
                                            state_id.with(0),
                                            parameters,
                                            &mut state.offset,
                                        )
                                    });
                                    field(ui, "Time Scale", |ui| {
                                        var_widget(
                                            ui,
                                            state_id.with(1),
                                            parameters,
                                            &mut state.time_scale,
                                        )
                                    });

                                    const DATA_NAMES: &'static [&'static str] = &["Clip", "Blend 1D", "Blend 2D"];
                                    let data_index = match &state.data {
                                        StateData::Clip { .. } => (0),
                                        StateData::Blend1D { .. } => (1),
                                        StateData::Blend2D { .. } => (2),
                                    };

                                    field(ui, "Data", |ui| {
                                        egui::ComboBox::from_id_source(state_id.with(2))
                                            .selected_text(DATA_NAMES[data_index])
                                            .show_ui(ui, |ui| {
                                                let checked = data_index == 0;
                                                if ui.selectable_label(checked, DATA_NAMES[0]).clicked() {
                                                    if !checked {
                                                        state.data = StateData::default_clip();
                                                    }
                                                }

                                                let checked = data_index == 0;
                                                if ui.selectable_label(checked, DATA_NAMES[0]).clicked() {
                                                    if !checked {
                                                        state.data = StateData::default_blend1d();
                                                    }
                                                }
                                                
                                                let checked = data_index == 0;
                                                if ui.selectable_label(checked, DATA_NAMES[0]).clicked() {
                                                    if !checked {
                                                        state.data = StateData::default_blend2d();
                                                    }
                                                }
                                            });
                                    });
                                    let data_id = state_id.with(3);
                                    match &mut state.data {
                                        StateData::Clip { clip } => {
                                        field(ui, "Clip", |ui| {
                                            clip_ref_widget(ui, data_id.with(0), &clips, &mut filters.clips, clip);
                                        });
                                        }
                                        StateData::Blend1D { value, blend } => {}
                                        StateData::Blend2D { mode, x, y, blend } => {}
                                    }

                                    ui.add_space(10.0);
                                    heading(ui, "Transitions");
                                }
                            }
                            Some(Selected::Transition(transition)) => {
                                heading(ui, "Transition");
                                let _ = transition;
                                ui.add_space(10.0);
                                heading(ui, "Others");
                            }
                            None => {
                                heading(ui, "Graph");
                                field(ui, "Name", |ui| {
                                    ui.text_edit_singleline(
                                        &mut target.mutate_without_modify().name,
                                    );
                                });

                                ui.add_space(10.0);
                                ui.horizontal(|ui| {
                                    heading(ui, "Parameters");

                                    let param_menu_id = id.with("param_menu");
                                    let mut response = ui.selectable_label(false, "➕");
                                    if response.clicked() {
                                        ui.memory().toggle_popup(param_menu_id);
                                    }
                                    response.rect.max.x = response.rect.min.x + 80.0;
                                    egui::popup::popup_below_widget(
                                        ui,
                                        param_menu_id,
                                        &response,
                                        |ui| {
                                            if ui.selectable_label(false, "Float").clicked() {
                                                let params = &mut target.mutate().parameters;
                                                let name = format!("Param{}", params.len());
                                                params.insert(name, Param::Float(0.0));
                                            }
                                            if ui.selectable_label(false, "Bool").clicked() {
                                                let params = &mut target.mutate().parameters;
                                                let name = format!("Param{}", params.len());
                                                params.insert(name, Param::Bool(false));
                                            }
                                        },
                                    );
                                });

                                enum ParamOp {
                                    None,
                                    Rename(String),
                                    Remove(String),
                                }
                                let mut modify = false;
                                let mut param_op = ParamOp::None;
                                let parameters = &mut target.mutate_without_modify().parameters;
                                for (param_id, name, param) in parameters.iter_mut() {
                                    ui.horizontal(|ui| {
                                        let x = ui.available_width();
                                        if label_editable(ui, id.with(param_id), name, temp_buffer)
                                        {
                                            param_op = ParamOp::Rename(name.clone());
                                        }
                                        ui.add_space((100.0 - x + ui.available_width()).max(0.0));

                                        match param {
                                            Param::Bool(v) => {
                                                let v0 = *v;
                                                ui.checkbox(v, String::default());
                                                modify |= v0 != *v;
                                            }
                                            Param::Float(v) => {
                                                let v0 = v.to_bits();
                                                ui.add(
                                                    DragValue::new(v).speed(0.01).max_decimals(3),
                                                );
                                                modify |= v0 != v.to_bits();
                                            }
                                        }

                                        if ui.selectable_label(false, "✖").clicked() {
                                            param_op = ParamOp::Remove(name.clone());
                                        }
                                    });
                                }
                                match param_op {
                                    ParamOp::None => {}
                                    ParamOp::Rename(name) => {
                                        target
                                            .mutate()
                                            .parameters
                                            .rename_by_name(&name, temp_buffer.clone());
                                    }
                                    ParamOp::Remove(name) => {
                                        target.mutate().parameters.remove_by_name(&name);
                                    }
                                }
                                defer_modify |= modify;

                                ui.add_space(10.0);
                                ui.horizontal(|ui| {
                                    heading(ui, "Layers");
                                    let response = ui.selectable_label(false, "➕");
                                    if response.clicked() {
                                        let layers = &mut target.mutate().layers;
                                        let layers_count = layers.len();
                                        layers.push(Layer {
                                            name: format!("Layer{}", layers_count),
                                            ..Default::default()
                                        });
                                    }
                                });

                                enum LayerOp {
                                    None,
                                    MoveUp(usize),
                                    MoveDown(usize),
                                    Remove(usize),
                                }
                                let mut layer_op = LayerOp::None;
                                let layers = &mut target.mutate_without_modify().layers;
                                let layers_count = layers.len();
                                for (layer_index, layer) in layers.iter_mut().enumerate() {
                                    ui.horizontal(|ui| {
                                        let x = ui.available_width();
                                        if label_editable(
                                            ui,
                                            id.with(layer_index),
                                            &layer.name,
                                            temp_buffer,
                                        ) {
                                            //modify = true;
                                            if !temp_buffer.is_empty() {
                                                layer.name.clone_from(temp_buffer);
                                            }
                                        }

                                        ui.add_space((100.0 - x + ui.available_width()).max(0.0));

                                        ui.add(
                                            DragValue::new(&mut layer.default_weight)
                                                .speed(0.01)
                                                .max_decimals(3),
                                        )
                                        .on_hover_text("Layer Weight");

                                        ui.checkbox(&mut layer.additive, String::default())
                                            .on_hover_text("Additive Mode");

                                        Frame::none().show(ui, |ui| {
                                            ui.set_enabled(layers_count > 1);
                                            if ui.selectable_label(false, "✖").clicked() {
                                                layer_op = LayerOp::Remove(layer_index);
                                            }
                                        });

                                        Frame::none().show(ui, |ui| {
                                            ui.set_enabled(layer_index < (layers_count - 1));
                                            if ui.selectable_label(false, "⬇").clicked() {
                                                layer_op = LayerOp::MoveDown(layer_index);
                                            }
                                        });

                                        Frame::none().show(ui, |ui| {
                                            ui.set_enabled(layer_index > 0);
                                            if ui.selectable_label(false, "⬆").clicked() {
                                                layer_op = LayerOp::MoveUp(layer_index);
                                            }
                                        });
                                    });
                                }
                                match layer_op {
                                    LayerOp::None => {}
                                    LayerOp::MoveUp(layer_index) => {
                                        let target_index = layer_index - 1;
                                        if *selected_layer == layer_index {
                                            *selected_layer = target_index;
                                        } else if *selected_layer == target_index {
                                            *selected_layer = layer_index;
                                        }
                                        target.mutate().layers.swap(target_index, layer_index);
                                    }
                                    LayerOp::MoveDown(layer_index) => {
                                        let target_index = layer_index + 1;
                                        if *selected_layer == layer_index {
                                            *selected_layer = target_index;
                                        } else if *selected_layer == target_index {
                                            *selected_layer = layer_index;
                                        }
                                        target.mutate().layers.swap(layer_index, target_index);
                                    }
                                    LayerOp::Remove(layer_index) => {
                                        target.mutate().layers.remove(layer_index);
                                        if layer_index == *selected_layer {
                                            if layer_index > 0 {
                                                *selected_layer -= 1;
                                            }
                                        }
                                    }
                                }
                                defer_modify |= modify;
                            }
                        });

                        let state_menu_id = id.with("state_menu");
                        let layer_menu_id = id.with("layer_menu");

                        let graph =
                            &mut target.mutate_without_modify().layers[*selected_layer].graph;

                        ui.set_clip_rect(rect);
                        draw_grid(ui, rect, *position, vec2(40.0, 40.0));

                        let position_offset = rect.min.to_vec2() + *position;

                        // Draw transitions
                        let mut transition_selection = None;
                        // Find transition
                        for source_index in 0..graph.node_count() {
                            let source_index = (source_index as u32).into();

                            let state = graph.node_weight_mut(source_index).unwrap();
                            let p0: Pos2 = state_pos(state) + position_offset;

                            for edge in graph.edges_directed(source_index, Outgoing) {
                                if let Some(target) = graph.node_weight(edge.target()) {
                                    let p1 = state_pos(target) + position_offset;
                                    let group = cache
                                        .grouped_transitions
                                        .entry((
                                            edge.source().index() as u32,
                                            edge.target().index() as u32,
                                        ))
                                        .or_insert_with(|| TransitionGroup {
                                            index: edge.id().index() as u32,
                                            count: 0,
                                            p0,
                                            p1,
                                        });

                                    group.count += 1;
                                    // NOTE: Done multiple times, but hopefully not as expensive
                                    group.p0 = p0;
                                    group.p1 = p1;
                                } else {
                                    // TODO: Remove invalid edge
                                }
                            }
                        }
                        // Draw transitions
                        for ((a, b), group) in &mut cache.grouped_transitions {
                            let TransitionGroup {
                                index,
                                count,
                                p0,
                                p1,
                            } = group;

                            let many = *count > 1;
                            *count = 0;

                            let current = Some(Selected::Transition(*index));
                            let is_selected = *selected == current;

                            if if *a == *b {
                                self_transition_widget(ui, *p0, many, is_selected)
                            } else {
                                transition_widget(ui, [*p0, *p1], many, is_selected)
                            } {
                                transition_selection = current;
                            }
                        }

                        // Draw adding transition helper
                        if let EditOp::AddTransition { state, position } = operation {
                            if let Some(state) = graph.node_weight_mut((*state).into()) {
                                let p0 = state_pos(state) + position_offset;
                                ui.painter().line_segment(
                                    [p0, *position],
                                    Stroke {
                                        width: 1.0,
                                        color: Color32::WHITE,
                                    },
                                );
                            } else {
                                // Exit
                                *operation = EditOp::None;
                            }
                        }

                        // Draw nodes
                        for index in 0..graph.node_count() {
                            let index = index as u32;
                            let state = graph.node_weight_mut(index.into()).unwrap();
                            let state_selection = Some(Selected::State(index as u32));

                            let center = state_pos(state) + position_offset;
                            let rect = Rect::from_center_size(center, STATE_SIZE);
                            let fill = if index == 0 {
                                Color32::from_rgb(200, 70, 0)
                            } else {
                                Color32::from_gray(60)
                            };

                            let response = state_widget(
                                ui,
                                rect,
                                &state.name,
                                fill,
                                state_selection == *selected,
                            );

                            if response.clicked() {
                                match operation {
                                    EditOp::AddTransition { state, .. } => {
                                        defer_modify = true;
                                        graph.add_edge(
                                            (*state).into(),
                                            index.into(),
                                            Transition::default(),
                                        );
                                        *operation = EditOp::None;
                                    }
                                    _ => {}
                                }
                                *selected = state_selection;
                            } else if response.secondary_clicked() {
                                *selected = state_selection;
                            } else if response.dragged_by(egui::PointerButton::Primary) {
                                // Drag node
                                let delta: [f32; 2] = response.drag_delta().into();
                                state.position += delta.into();
                            }

                            // Clear transition selection
                            if response.hovered() {
                                transition_selection = None;
                            }

                            // Open state menu
                            if let Some(cursor_position) = response.hover_pos() {
                                if response.secondary_clicked()
                                    && !ui.memory().is_popup_open(state_menu_id)
                                {
                                    *context_menu_position = cursor_position;
                                    ui.memory().open_popup(state_menu_id);
                                }
                            }
                        }
                        std::mem::drop(graph);

                        let mut response = ui.interact(rect, id, egui::Sense::click_and_drag());
                        if response.dragged_by(egui::PointerButton::Middle) {
                            // Pan
                            *position += response.drag_delta();
                        } else {
                            if let Some(cursor_position) = response.hover_pos() {
                                if response.secondary_clicked()
                                    && !ui.memory().is_popup_open(layer_menu_id)
                                {
                                    *context_menu_position = cursor_position;
                                    ui.memory().open_popup(layer_menu_id);
                                }
                            }
                        }

                        // Handle selection
                        if response.clicked() {
                            if transition_selection.is_some() {
                                *selected = transition_selection;
                            } else {
                                // Deselect
                                *selected = None;
                            }
                        }

                        response.rect =
                            egui::Rect::from_min_size(*context_menu_position, (150.0, 1.0).into());

                        egui::popup::popup_below_widget(ui, layer_menu_id, &response, |ui| {
                            if ui.selectable_label(false, "Create State").clicked() {
                                let position = *context_menu_position - *position - rect.min;
                                let position: [f32; 2] = position.into();

                                target.mutate().layers[*selected_layer]
                                    .graph
                                    .add_node(State {
                                        name: "Empty".to_string(),
                                        position: position.into(),
                                        ..Default::default()
                                    });
                            }
                        });

                        let mut delete_selected = false;
                        egui::popup::popup_below_widget(ui, state_menu_id, &response, |ui| {
                            //ui.set_enabled(true);
                            match *selected {
                                Some(Selected::State(state)) => {
                                    let response = ui.selectable_label(false, "Create Transition");
                                    if response.clicked() {
                                        if let Some(position) = response.interact_pointer_pos() {
                                            *operation = EditOp::AddTransition { state, position };
                                        }
                                    }

                                    if ui.selectable_label(state == 0, "Set as Default").clicked() {
                                        target.mutate().layers[*selected_layer]
                                            .set_default_state(state);
                                        *selected = None;
                                    }

                                    if ui.selectable_label(false, "Delete").clicked() {
                                        delete_selected = true;
                                    }
                                }
                                _ => {}
                            }
                        });

                        // Graph events (cursor needs to stay inside the graph)
                        if let Some(pos) = ui.input().pointer.hover_pos() {
                            if rect.contains(pos) {
                                // Visual feedback of graph editing

                                ui.painter().rect_stroke(
                                    Rect {
                                        min: rect.min + vec2(0.5, 0.5),
                                        max: rect.max - vec2(0.5, 0.5),
                                    },
                                    0.0,
                                    Stroke {
                                        width: 0.5,
                                        color: Color32::from_gray(100),
                                    },
                                );

                                // Update operation
                                match operation {
                                    EditOp::AddTransition { position, .. } => {
                                        if let Some(p) = response.hover_pos() {
                                            *position = p;
                                        }
                                        if ui.input().key_pressed(Key::Escape) {
                                            *operation = EditOp::None;
                                        }
                                    }
                                    _ => {}
                                }

                                // Delete selection
                                if delete_selected || ui.input().key_pressed(Key::Delete) {
                                    match selected {
                                        Some(Selected::State(index)) => {
                                            target.mutate().layers[*selected_layer]
                                                .graph
                                                .remove_node((*index).into());
                                        }
                                        Some(Selected::Transition(index)) => {
                                            target.mutate().layers[*selected_layer]
                                                .graph
                                                .remove_edge((*index).into());
                                        }
                                        _ => {}
                                    }
                                    *selected = None;
                                }
                            }
                        }

                        // Force graph modification
                        if defer_modify {
                            target.mutate();
                        }
                    }

                    // Clear cache when the graph is modified
                    if target.is_modified() {
                        cache.clear();
                    }
                } else {
                    *editing = None;
                }
            } else {
                // No graph is selected for editing
                let mut rect = rect;
                let offset = rect.size().y * 0.5 - 20.0;
                rect.min.y += offset;
                rect.max.y -= offset;
                ui.allocate_ui_at_rect(rect, |ui: &mut Ui| {
                    ui.vertical_centered(|ui| {
                        ui.label("No graph is currently been edited");
                        ui.add_space(5.0);
                        ui.horizontal(|ui| {
                            ui.add_space(ui.available_width() * 0.5 - 67.0);
                            if ui.button("Create Graph").clicked() {
                                menu_op = MenuOp::CreateNew;
                            }
                            if ui.button("Load").clicked() {
                                menu_op = MenuOp::Load;
                            }
                        });
                    });
                });
            }
        });

    match menu_op {
        MenuOp::None => {}
        MenuOp::CreateNew => {
            // Create a new graph and select it
            let name = format!("Graph{}", graphs.len());
            *editing = Some(graphs.add(AnimaGraph {
                name,
                ..Default::default()
            }));
        }
        MenuOp::Load => {
            let load_path = native_dialog::FileDialog::new()
                .add_filter("AnimaGraph", EXTENSIONS)
                .show_open_single_file();

            if let Ok(Some(load_path)) = load_path {
                // TOOD: Log error
                match load_file(load_path.as_path(), &asset_server, &mut graphs) {
                    Ok(loaded) => {
                        *editing = Some(loaded);
                    }
                    Err(err) => {
                        error!(
                            "`AnimaGraph` at '{}' failed to load, reason: {}",
                            load_path.display(),
                            err
                        );
                    }
                }
            }
        }
    }
}

fn save_file(save_path: &Path, asset_server: &AssetServer, graph: &AnimaGraph) -> Result<()> {
    let file = File::create(save_path)?;

    let mut serializer = ron::Serializer::new(file, Some(ron::ser::PrettyConfig::default()), true)?;
    // Oh for fuck sake, rust failed to provide a meaningful error here, remove the `&mut` and see ... avenge me future me report this error
    asset_server.serialize_with_asset_refs(&mut serializer, graph)?;
    Ok(())
}

fn load_file(
    save_path: &Path,
    asset_server: &AssetServer,
    assets: &mut Assets<AnimaGraph>,
) -> Result<Handle<AnimaGraph>> {
    let mut file = File::open(save_path)?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)?;

    let mut deserializer = ron::de::Deserializer::from_bytes(bytes.as_slice())?;
    let graph = asset_server.deserialize_with_asset_refs(&mut deserializer)?;

    Ok(assets.add(graph))
}

#[inline]
fn heading(ui: &mut Ui, label: impl ToString) {
    ui.add(Label::new(label).text_style(TextStyle::Button));
}

#[inline]
fn field<T>(ui: &mut Ui, label: impl Into<Label>, add_contents: impl FnOnce(&mut Ui) -> T) -> T {
    ui.horizontal(|ui| {
        let x = ui.available_width();
        ui.label(label);
        ui.add_space((100.0 - x + ui.available_width()).max(0.0));
        (add_contents)(ui)
    })
    .inner
}

#[inline]
fn param_widget(
    ui: &mut Ui,
    id: impl Into<Id>,
    params: &Parameters,
    param_id: &mut ParamId,
    param_type: VarType,
) -> Response {
    ui.horizontal(|ui| {
        let param = params.get_by_id(*param_id);
        let name = match (param_type, param) {
            (VarType::Bool, Some((name, Param::Bool(_)))) => name.as_str(),
            (VarType::Float, Some((name, Param::Float(_)))) => name.as_str(),
            _ => "",
        };
        egui::ComboBox::from_id_source(id.into())
            .selected_text(name)
            .show_ui(ui, |ui| {
                for (other_id, other_name, param) in params.iter() {
                    match (param_type, param) {
                        (VarType::Bool, Param::Bool(_)) | (VarType::Float, Param::Float(_)) => {
                            if ui
                                .selectable_label(other_id == *param_id, other_name)
                                .clicked()
                            {
                                *param_id = other_id;
                            }
                        }
                        _ => {}
                    }
                }
            })
    })
    .inner
}

trait TypeWidget {
    fn widget(&mut self, ui: &mut Ui) -> Response;
    fn var_type() -> VarType;
}

impl TypeWidget for bool {
    #[inline]
    fn widget(&mut self, ui: &mut Ui) -> Response {
        ui.checkbox(self, String::default())
    }

    #[inline]
    fn var_type() -> VarType {
        VarType::Bool
    }
}

impl TypeWidget for f32 {
    #[inline]
    fn widget(&mut self, ui: &mut Ui) -> Response {
        ui.add(DragValue::new(self).speed(0.01).max_decimals(3))
    }

    #[inline]
    fn var_type() -> VarType {
        VarType::Float
    }
}

#[inline]
fn var_widget<T: Default + TypeWidget>(
    ui: &mut Ui,
    id: impl Into<Id>,
    params: &Parameters,
    var: &mut Var<T>,
) -> Response {
    ui.horizontal(|ui| {
        ui.checkbox(&mut var.use_param, "🗝");
        if var.use_param {
            param_widget(ui, id, params, &mut var.param, T::var_type());
        } else {
            var.value.widget(ui);
        }
    })
    .response
}

#[inline]
fn asset_ref_widget<T: Asset>(
    ui: &mut Ui,
    id: impl Into<Id>,
    assets: &Assets<T>,
    filter: &mut String,
    asset_ref: &mut AssetRef<T>,
    asset_name: impl Fn(&T) -> &str
) -> Response {
    ui.horizontal(|ui| {
        let handle: Handle<T> = asset_ref.clone().into();
        egui::ComboBox::from_id_source(id.into())
            .selected_text(assets.get(&handle).map(&asset_name).unwrap_or(""))
            .show_ui(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label("🔎");
                    ui.text_edit_singleline(filter);
                    if ui.selectable_label(false, "✖").clicked() {
                        filter.clear();
                    }
                });
        
                ui.separator();

                let filter = filter.as_str();
                egui::ScrollArea::auto_sized().show(ui, |ui| {
                    for (handle_id, asset) in assets.iter() {
                        let name = asset_name(asset);
                        if filter.len() > 0 && !name.contains(filter) {
                            continue;
                        }
                        
                        if ui.selectable_label(handle_id == handle.id, name).clicked() {
                            *asset_ref = assets.get_handle(handle_id).into();
                        }
                    }
                });
            })
    })
    .response
}

#[inline]
fn clip_ref_widget(
    ui: &mut Ui,
    id: impl Into<Id>,
    assets: &Assets<Clip>,
    filter: &mut String,
    asset_ref: &mut AssetRef<Clip>,
) -> Response {
    asset_ref_widget(ui, id, assets, filter, asset_ref, |clip| clip.name.as_str())
}


// fn right_to_left<T>(ui: &mut Ui, add_contents: impl FnOnce(&mut Ui) -> T) -> T {
//     ui.allocate_ui_with_layout(
//         vec2(ui.available_width(), 0.0),
//         Layout::right_to_left(),
//         add_contents,
//     )
//     .inner
// }

#[inline]
fn label_editable(ui: &mut Ui, id: impl Into<Id>, text: &String, temp_buffer: &mut String) -> bool {
    let id = id.into();
    if ui.memory().is_popup_open(id) {
        let response = ui.add(TextEdit::singleline(temp_buffer).desired_width(80.0));
        if response.lost_focus()
            || response.clicked_elsewhere()
            || ui.input().key_pressed(Key::Escape)
        {
            ui.memory().close_popup();
            temp_buffer != text
        } else {
            false
        }
    } else {
        if ui
            .add(Label::new(text).sense(Sense::click()))
            .double_clicked()
        {
            ui.memory().open_popup(id);
            temp_buffer.clone_from(text);
        }
        false
    }
}

#[inline]
fn state_pos(state: &State) -> Pos2 {
    let position: [f32; 2] = state.position.into();
    Pos2::from(position)
}

#[inline]
fn state_widget(
    ui: &mut Ui,
    rect: Rect,
    name: impl Into<String>,
    fill: Color32,
    selected: bool,
) -> Response {
    let mut ui = ui.child_ui(rect, *ui.layout());

    let button_padding = ui.spacing().button_padding;
    let galley = ui.fonts().layout_no_wrap(TextStyle::Button, name.into());

    let (rect, response) = ui.allocate_at_least(rect.size(), Sense::click_and_drag());
    response.widget_info(|| WidgetInfo::labeled(WidgetType::Button, &galley.text));

    if ui.clip_rect().intersects(rect) {
        let visuals = if !selected {
            ui.style().interact(&response)
        } else {
            &ui.style().visuals.widgets.active
        };

        let text_cursor = Layout::centered_and_justified(egui::Direction::TopDown)
            .align_size_within_rect(galley.size, rect.shrink2(button_padding))
            .min;

        ui.painter().rect(
            rect.expand(visuals.expansion),
            visuals.corner_radius,
            fill,
            visuals.bg_stroke,
        );

        let text_color = visuals.text_color();
        ui.painter().galley(text_cursor, galley, text_color);
    }

    response
}

#[inline]
fn transition_widget(ui: &mut Ui, line: [Pos2; 2], many: bool, selected: bool) -> bool {
    let [mut p0, mut p1] = line;

    let v = p1 - p0;
    let m = v.length();
    let v_normalized = v / m.max(1e-12);
    let n = vec2(-v_normalized.y, v_normalized.x);
    let lane_offset = n * 12.0;
    p0 += lane_offset;
    p1 += lane_offset;

    let hover = if let Some(pos) = ui.input().pointer.hover_pos() {
        let u = pos - p0;
        let n = u.x * v_normalized.x + u.y * v_normalized.y;
        let d = u - n.clamp(0.0, m) * v_normalized;
        d.length() <= 8.0
    } else {
        false
    };

    let stroke = if selected {
        Stroke {
            width: 2.0,
            color: Color32::LIGHT_GRAY,
        }
    } else if hover {
        Stroke {
            width: 2.0,
            color: Color32::GRAY,
        }
    } else {
        Stroke {
            width: 1.0,
            color: Color32::GRAY,
        }
    };

    ui.painter().line_segment([p0, p1], stroke);

    let right = n * ARROW_SIZE;
    let forward = v_normalized * ARROW_SIZE;
    let mut t0 = p0 + (v * 0.5);
    let mut t1 = t0 - forward;
    let mut t2 = t1 - right;
    t1 += right;

    draw_triangle(ui, t0, t1, t2, stroke);

    if many {
        t0 += forward;
        t1 += forward;
        t2 += forward;
        draw_triangle(ui, t0, t1, t2, stroke);

        let back = forward * 2.0;
        t0 -= back;
        t1 -= back;
        t2 -= back;
        draw_triangle(ui, t0, t1, t2, stroke);
    }

    hover
}

#[inline]
fn self_transition_widget(ui: &mut Ui, pos: Pos2, many: bool, selected: bool) -> bool {
    let center = pos - STATE_SIZE * 0.5;
    let radius = vec2(20.0, 20.0);
    let area = Rect {
        min: center - radius,
        max: center + radius,
    };

    let hover = ui
        .input()
        .pointer
        .hover_pos()
        .map_or(false, |pos| area.contains(pos));

    let stroke = if selected {
        Stroke {
            width: 2.0,
            color: Color32::LIGHT_GRAY,
        }
    } else if hover {
        Stroke {
            width: 2.0,
            color: Color32::GRAY,
        }
    } else {
        Stroke {
            width: 1.0,
            color: Color32::GRAY,
        }
    };

    ui.painter().circle_stroke(center, 20.0, stroke);

    const DIAG: Vec2 = vec2(0.707106781, 0.707106781);
    const NORMAL: Vec2 = vec2(-0.707106781, 0.707106781);

    let right = DIAG * ARROW_SIZE;
    let forward = NORMAL * ARROW_SIZE;
    let mut t0 = center - (DIAG * 20.0);
    let mut t1 = t0 - forward;
    let mut t2 = t1 - right;
    t1 += right;

    draw_triangle(ui, t0, t1, t2, stroke);

    if many {
        t0 += forward;
        t1 += forward;
        t2 += forward;
        draw_triangle(ui, t0, t1, t2, stroke);
    }

    hover
}

#[inline]
fn draw_triangle(ui: &Ui, t0: Pos2, t1: Pos2, t2: Pos2, stroke: impl Into<Stroke>) {
    let stroke = stroke.into();
    ui.painter().line_segment([t0, t1], stroke);
    ui.painter().line_segment([t1, t2], stroke);
    ui.painter().line_segment([t2, t0], stroke);
}

#[inline]
fn draw_grid(ui: &Ui, mut rect: Rect, offset: Vec2, size: Vec2) {
    let Rect { min, mut max } = rect;
    max -= vec2(EPSILON, EPSILON);

    let offset = vec2(offset.x.rem_euclid(size.x), offset.y.rem_euclid(size.y));
    let mut diag = min + offset - size;
    let stroke = Stroke {
        width: 0.2,
        color: Color32::from_gray(80),
    };

    let p = ui.painter();
    while diag.x < max.x || diag.y < max.y {
        p.line_segment([pos2(diag.x, min.y), pos2(diag.x, max.y)], stroke);
        p.line_segment([pos2(min.x, diag.y), pos2(max.x, diag.y)], stroke);
        diag += size;
    }

    rect.min += vec2(0.2, 0.2);
    rect.max -= vec2(0.2, 0.2);
    p.rect_stroke(rect, 0.0, stroke);
}

fn action_needed(
    ui: &mut Ui,
    rect: Rect,
    message: impl Into<Label>,
    action_label: impl ToString,
) -> Response {
    // No graph is selected for editing
    let mut rect = rect;
    let offset = rect.size().y * 0.5 - 20.0;
    rect.min.y += offset;
    rect.max.y -= offset;
    ui.allocate_ui_at_rect(rect, |ui: &mut Ui| {
        ui.vertical_centered(|ui| {
            ui.label(message);
            ui.add_space(5.0);
            ui.button(action_label)
        })
        .inner
    })
    .inner
}

///////////////////////////////////////////////////////////////////////////////

/// Helper struct to allow mutate or view an asset without raising the [`AssetEvent::Modified`] event,
/// asset events will be raised (or not) on drop;
pub struct Modify<'a, T: Asset> {
    assets: &'a mut Assets<T>,
    target: &'a mut T,
    handle_id: HandleId,
    modified: bool,
}

impl<'a, T: Asset> Modify<'a, T> {
    pub fn new(assets: &'a mut Assets<T>, handle: impl Into<HandleId>) -> Option<Self> {
        let handle_id = handle.into();
        if let Some(target) = assets.get(handle_id) {
            // SAFETY: We currently mutable ownership over the assets,
            // this is just to avoids unneeded asset modified events
            let target = unsafe { &mut *(target as *const _ as *mut _) };
            Some(Self {
                assets,
                target,
                handle_id,
                modified: false,
            })
        } else {
            None
        }
    }

    pub fn is_modified(&self) -> bool {
        self.modified
    }

    #[inline(always)]
    pub fn view(&self) -> &T {
        self.target
    }

    #[inline(always)]
    pub fn mutate(&mut self) -> &mut T {
        self.modified = true;
        self.target
    }

    #[inline(always)]
    pub fn mutate_without_modify(&mut self) -> &mut T {
        self.target
    }
}

impl<'a, T: Asset> std::ops::Drop for Modify<'a, T> {
    fn drop(&mut self) {
        if self.modified {
            //  Trigger an asset modify event
            self.assets.get_mut(self.handle_id);
        }
    }
}

///////////////////////////////////////////////////////////////////////////////

pub struct AnimatorControllerEditorPlugin;

impl Plugin for AnimatorControllerEditorPlugin {
    fn build(&self, app: &mut bevy::prelude::AppBuilder) {
        app.insert_resource(AnimaGraphEditor {
            open: true,
            editing: None,
            live: None,
            operation: EditOp::None,
            position: Vec2::ZERO,
            selected: None,
            selected_layer: 0,
            context_menu_position: Pos2::ZERO,
            temp_buffer: String::default(),
            cache: Cache::default(),
            filters: Filters::default(),
        })
        .add_system_to_stage(CoreStage::PostUpdate, animator_graph_editor_system.system());
    }
}
