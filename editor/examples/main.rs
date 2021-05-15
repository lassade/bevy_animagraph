use std::f32::EPSILON;

use bevy::prelude::*;
use bevy_animagraph::AnimatorControllerPlugin;
use bevy_animagraph_editor::AnimatorControllerEditorPlugin;
use bevy_egui::EguiPlugin;

fn main() {
    App::build()
        .add_plugins(DefaultPlugins)
        .add_plugin(EguiPlugin)
        .add_plugin(AnimatorControllerPlugin)
        .add_plugin(AnimatorControllerEditorPlugin)
        .run();
}
