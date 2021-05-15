use std::f32::EPSILON;

use bevy::{
    asset::{Asset, HandleId},
    prelude::*,
    utils::HashMap,
};
use bevy_animagraph::{
    petgraph::{visit::EdgeRef, EdgeDirection::Outgoing},
    Animagraph, Layer, State, Transition,
};
use bevy_egui::{
    egui::{
        self, emath::NumExt, pos2, vec2, Align, Button, Color32, Key, Label, Layout, PointerButton,
        Pos2, Rect, Response, Sense, Stroke, TextStyle, Ui, Vec2, WidgetInfo, WidgetType,
    },
    EguiContext,
};

///////////////////////////////////////////////////////////////////////////////

#[derive(Debug, PartialEq, Eq, Copy, Clone)]
enum Selected {
    State(u32),
    Transition(u32),
}

#[derive(Debug, PartialEq, Eq, Copy, Clone)]
enum EditorOperation {
    None,
    AddingTransition { state: u32, position: Pos2 },
}

struct TransitionGroup {
    index: u32,
    count: usize,
    p0: Pos2,
    p1: Pos2,
}

#[derive(Default)]
struct Cache {
    target: Option<Handle<Animagraph>>,
    grouped_transitions: HashMap<(u32, u32), TransitionGroup>,
}

impl Cache {
    fn clear(&mut self) {
        self.target = None;
        self.grouped_transitions.clear();
    }
}

struct AnimagraphEditor {
    open: bool,
    editing: Option<Handle<Animagraph>>,
    operation: EditorOperation,
    position: Vec2,
    selected: Option<Selected>,
    selected_layer: usize,
    context_menu_position: Pos2,
    cache: Cache,
}

const STATE_SIZE: Vec2 = vec2(180.0, 40.0);
const ARROW_SIZE: f32 = 8.0;

fn animator_graph_editor_system(
    mut graph_editor: ResMut<AnimagraphEditor>,
    mut graphs: ResMut<Assets<Animagraph>>,
    egui_context: Res<EguiContext>,
) {
    let AnimagraphEditor {
        open,
        editing,
        operation,
        position,
        selected,
        selected_layer,
        context_menu_position,
        cache,
    } = &mut *graph_editor;

    // Clear cache
    if cache.target != *editing {
        cache.clear();
        cache.target = editing.clone();
    }

    egui::Window::new("Animagraph Editor")
        .default_size([800.0, 600.0])
        .open(open)
        .show(egui_context.ctx(), |ui| {
            //let rect = ui.available_rect_before_wrap();
            let (id, rect) = ui.allocate_space(ui.available_size());

            if let Some(graph_handle) = editing {
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
                        // Tool bar
                        let mut toolbar_rect = rect;
                        toolbar_rect.max.y = toolbar_rect.min.y + 20.0;
                        ui.allocate_ui_at_rect(toolbar_rect, |ui: &mut Ui| {
                            ui.horizontal(|ui| {
                                // Graph name
                                ui.add(
                                    egui::TextEdit::singleline(
                                        &mut target.mutate_without_modify().name,
                                    )
                                    .desired_width(100.0)
                                    .hint_text("Name"),
                                );
                                // Select active layer
                                egui::ComboBox::from_id_source("layer_select")
                                    .selected_text(&target.view().layers[*selected_layer].name)
                                    .show_ui(ui, |ui| {
                                        for i in 0..layer_count {
                                            let text = &target.view().layers[i].name;
                                            ui.selectable_value(selected_layer, i, text);
                                        }
                                    });
                            });
                        });

                        let state_menu_id = id.with("state_menu");
                        let layer_menu_id = id.with("layer_menu");

                        let mut modify = false;
                        let graph =
                            &mut target.mutate_without_modify().layers[*selected_layer].graph;

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
                        if let EditorOperation::AddingTransition { state, position } = operation {
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
                                *operation = EditorOperation::None;
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
                                    EditorOperation::AddingTransition { state, .. } => {
                                        modify = true;
                                        graph.add_edge(
                                            (*state).into(),
                                            index.into(),
                                            Transition::default(),
                                        );
                                        *operation = EditorOperation::None;
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
                        if modify {
                            target.mutate();
                        }

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
                                            *operation = EditorOperation::AddingTransition {
                                                state,
                                                position,
                                            };
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

                        // Update operation
                        match operation {
                            EditorOperation::AddingTransition { position, .. } => {
                                if let Some(p) = response.hover_pos() {
                                    *position = p;
                                }
                            }
                            _ => {}
                        }

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
                let response = action_needed(
                    ui,
                    rect,
                    "No graph is currently been edited",
                    "Create Graph",
                );
                if response.clicked() {
                    *editing = Some(graphs.add(Animagraph::default()));
                }
            }
        });
}

#[inline(always)]
fn state_pos(state: &State) -> Pos2 {
    let position: [f32; 2] = state.position.into();
    Pos2::from(position)
}

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

        let text_cursor = ui
            .layout()
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

fn draw_grid(ui: &Ui, rect: Rect, offset: Vec2, size: Vec2) {
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
}

fn action_needed(
    ui: &mut Ui,
    rect: Rect,
    message: impl Into<Label>,
    action_label: impl Into<String>,
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
        app.insert_resource(AnimagraphEditor {
            open: true,
            editing: None,
            operation: EditorOperation::None,
            position: Vec2::ZERO,
            selected: None,
            selected_layer: 0,
            context_menu_position: Pos2::ZERO,
            cache: Cache::default(),
        })
        .add_system_to_stage(CoreStage::PostUpdate, animator_graph_editor_system.system());
    }
}
