use std::f32::EPSILON;

use bevy::prelude::*;
use bevy_animation_controller::{AnimatorControllerPlugin, AnimatorGraph};
use bevy_egui::{
    egui::{self, pos2, vec2, Color32, Pos2, Rect, Stroke, Ui, Vec2},
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
    Node(u32),
}

struct AnimatorGraphEditor {
    open: bool,
    editing: Option<Handle<AnimatorGraph>>,
    position: Pos2,
    selected: Option<Selected>,
}

fn animator_graph_editor_system(
    mut animator_graph_editor: ResMut<AnimatorGraphEditor>,
    egui_context: Res<EguiContext>,
) {
    let AnimatorGraphEditor {
        open,
        editing,
        position,
        selected,
    } = &mut *animator_graph_editor;

    egui::Window::new("Animator Graph Editor")
        .default_size([800.0, 600.0])
        .open(open)
        .show(egui_context.ctx(), |ui| {
            let (id, rect) = ui.allocate_space(ui.available_size());

            let response = ui.interact(rect, id, egui::Sense::click_and_drag());
            if response.dragged_by(egui::PointerButton::Middle) {
                // Pan
                *position += response.drag_delta();
            }

            grid(ui, rect, *position, vec2(20.0, 20.0));

            // TODO: No graph selected
            // TODO: Draw nodes
        });
}

#[inline]
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

pub struct AnimatorControllerEditorPlugin;

impl Plugin for AnimatorControllerEditorPlugin {
    fn build(&self, app: &mut bevy::prelude::AppBuilder) {
        app.insert_resource(AnimatorGraphEditor {
            open: true,
            editing: None,
            position: Pos2::ZERO,
            selected: None,
        })
        .add_system_to_stage(CoreStage::PostUpdate, animator_graph_editor_system.system());
    }
}
