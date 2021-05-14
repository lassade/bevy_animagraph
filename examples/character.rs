use std::f32::EPSILON;

use bevy::{
    asset::{Asset, HandleId},
    prelude::*,
};
use bevy_animation_controller::{
    AnimatorControllerPlugin, AnimatorGraph, Layer, State, Transition,
};
use bevy_egui::{
    egui::{
        self, emath::NumExt, pos2, vec2, Align, Button, Color32, Key, Label, Layout, Pos2, Rect,
        Response, Sense, Stroke, TextStyle, Ui, Vec2, WidgetInfo, WidgetType,
    },
    EguiContext, EguiPlugin,
};

fn main() {
    App::build()
        .add_plugins(DefaultPlugins)
        .add_plugin(EguiPlugin)
        .add_plugin(AnimatorControllerPlugin)
        .add_plugin(AnimatorControllerEditorPlugin)
        .run();
}

///////////////////////////////////////////////////////////////////////////////

#[derive(Debug, PartialEq, Eq)]
enum Selected {
    State(u32),
    Transition(u32, u32),
}

struct AnimatorGraphEditor {
    open: bool,
    editing: Option<Handle<AnimatorGraph>>,
    position: Vec2,
    selected: Option<Selected>,
    selected_layer: usize,
    context_menu_position: Pos2,
}

fn animator_graph_editor_system(
    mut animator_graph_editor: ResMut<AnimatorGraphEditor>,
    mut animator_graphs: ResMut<Assets<AnimatorGraph>>,
    egui_context: Res<EguiContext>,
) {
    let AnimatorGraphEditor {
        open,
        editing,
        position,
        selected,
        selected_layer,
        context_menu_position,
    } = &mut *animator_graph_editor;

    egui::Window::new("Animator Graph Editor")
        .default_size([800.0, 600.0])
        .open(open)
        .show(egui_context.ctx(), |ui| {
            //let rect = ui.available_rect_before_wrap();
            let (id, rect) = ui.allocate_space(ui.available_size());

            if let Some(graph_handle) = editing {
                if let Some(mut graph) = Modify::new(&mut *animator_graphs, &*graph_handle) {
                    let layer_count = graph.view().layers.len();
                    if *selected_layer > layer_count {
                        *selected_layer = 0;
                    }

                    if layer_count == 0 {
                        // Create the default layer
                        if action_needed(ui, rect, "Graph have no layers", "Create Layer").clicked()
                        {
                            graph.mutate().layers.push(Layer {
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
                                        &mut graph.mutate_without_modify().name,
                                    )
                                    .desired_width(100.0)
                                    .hint_text("Name"),
                                );
                                // Select active layer
                                egui::ComboBox::from_id_source("layer_select")
                                    .selected_text(&graph.view().layers[*selected_layer].name)
                                    .show_ui(ui, |ui| {
                                        for i in 0..layer_count {
                                            let text = &graph.view().layers[i].name;
                                            ui.selectable_value(selected_layer, i, text);
                                        }
                                    });
                            });
                        });

                        draw_grid(ui, rect, *position, vec2(40.0, 40.0));

                        // Draw nodes
                        let state_menu_id = id.with("state_menu");
                        for (index, node) in graph.view().layers[*selected_layer]
                            .graph
                            .raw_nodes()
                            .iter()
                            .enumerate()
                        {
                            let state = &node.weight;
                            let selection_index = Some(Selected::State(index as u32));

                            let center: [f32; 2] = state.position.into();
                            let center = rect.min + (Vec2::from(center) + *position);
                            let rect = Rect::from_center_size(center, vec2(180.0, 40.0));
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
                                selection_index == *selected,
                            );

                            if response.clicked() || response.secondary_clicked() {
                                *selected = selection_index;
                            }

                            if let Some(cursor_position) = response.hover_pos() {
                                if response.secondary_clicked()
                                    && !ui.memory().is_popup_open(state_menu_id)
                                {
                                    *context_menu_position = cursor_position;
                                    ui.memory().open_popup(state_menu_id);
                                }
                            }
                        }

                        let layer_menu_id = id.with("layer_menu");
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

                        response.rect =
                            egui::Rect::from_min_size(*context_menu_position, (150.0, 1.0).into());

                        egui::popup::popup_below_widget(ui, layer_menu_id, &response, |ui| {
                            if ui.selectable_label(false, "Create State").clicked() {
                                let position = *context_menu_position - *position - rect.min;
                                let position: [f32; 2] = position.into();

                                graph.mutate().layers[*selected_layer]
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
                            if ui.selectable_label(false, "Create Transition").clicked() {}
                            if ui.selectable_label(false, "Delete").clicked() {
                                delete_selected = true;
                            }
                        });

                        if delete_selected || ui.input().key_pressed(Key::Delete) {
                            match selected {
                                Some(Selected::State(index)) => {
                                    graph.mutate().layers[*selected_layer]
                                        .graph
                                        .remove_node((*index).into());
                                }
                                _ => {}
                            }
                            *selected = None;
                        }
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
                    *editing = Some(animator_graphs.add(AnimatorGraph::default()));
                }
            }
        });
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
        app.insert_resource(AnimatorGraphEditor {
            open: true,
            editing: None,
            position: Vec2::ZERO,
            selected: None,
            selected_layer: 0,
            context_menu_position: Pos2::ZERO,
        })
        .add_system_to_stage(CoreStage::PostUpdate, animator_graph_editor_system.system());
    }
}
