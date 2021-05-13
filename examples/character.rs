use std::f32::EPSILON;

use bevy::prelude::*;
use bevy_animation_controller::{AnimatorControllerPlugin, AnimatorGraph, Layer};
use bevy_egui::{
    egui::{self, pos2, vec2, Color32, Label, Pos2, Rect, Response, Stroke, Ui, Vec2},
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

enum Selected {
    State(u32),
    Transition(u32),
}

struct AnimatorGraphEditor {
    open: bool,
    editing: Option<Handle<AnimatorGraph>>,
    position: Pos2,
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
                if let Some(graph) = animator_graphs.get(&*graph_handle) {
                    let layer_count = graph.layers.len();
                    if *selected_layer > layer_count {
                        *selected_layer = 0;
                    }

                    if layer_count == 0 {
                        // Create the default layer
                        if action_needed(ui, rect, "Graph have no layers", "Create Layer").clicked()
                        {
                            let graph = animator_graphs.get_mut(&*graph_handle).unwrap();
                            graph.layers.push(Layer {
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
                                    // SAFETY: We currently mutable ownership over the graph assets,
                                    // this is just to avoids unneeded asset modified events
                                    egui::TextEdit::singleline(unsafe {
                                        &mut *(&graph.name as *const _ as *mut _)
                                    })
                                    .desired_width(100.0)
                                    .hint_text("Name"),
                                );
                                // Select active layer
                                egui::ComboBox::from_id_source("layer_select")
                                    .selected_text(&graph.layers[*selected_layer].name)
                                    .show_ui(ui, |ui| {
                                        for i in 0..layer_count {
                                            let text = &graph.layers[i].name;
                                            ui.selectable_value(selected_layer, i, text);
                                        }
                                    });
                            });
                        });

                        grid(ui, rect, *position, vec2(40.0, 40.0));
                        // TODO: Draw nodes

                        let mut response = ui.interact(rect, id, egui::Sense::click_and_drag());
                        let context_menu_id = id.with("context_menu");

                        if response.dragged_by(egui::PointerButton::Middle) {
                            // Pan
                            *position += response.drag_delta();
                        } else {
                            if let Some(cursor_position) = response.hover_pos() {
                                if response.secondary_clicked()
                                    && !ui.memory().is_popup_open(context_menu_id)
                                {
                                    *context_menu_position = cursor_position;
                                    ui.memory().open_popup(context_menu_id);
                                }
                            }
                        }

                        response.rect =
                            egui::Rect::from_min_size(*context_menu_position, (150.0, 1.0).into());

                        egui::popup::popup_below_widget(ui, context_menu_id, &response, |ui| {
                            //ui.set_enabled(selected);
                            if ui.selectable_label(false, "Create State").clicked() {
                                //curve.set_interpolation(index, Interpolation::Step);
                            }
                        });
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

fn grid(ui: &Ui, rect: Rect, offset: Pos2, size: Vec2) {
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

pub struct AnimatorControllerEditorPlugin;

impl Plugin for AnimatorControllerEditorPlugin {
    fn build(&self, app: &mut bevy::prelude::AppBuilder) {
        app.insert_resource(AnimatorGraphEditor {
            open: true,
            editing: None,
            position: Pos2::ZERO,
            selected: None,
            selected_layer: 0,
            context_menu_position: Pos2::ZERO,
        })
        .add_system_to_stage(CoreStage::PostUpdate, animator_graph_editor_system.system());
    }
}
