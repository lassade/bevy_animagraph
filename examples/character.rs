use bevy::prelude::*;
use bevy_animation_controller::AnimatorControllerPlugin;
use bevy_egui::{egui, EguiContext, EguiPlugin};

fn main() {
    App::build()
        .add_plugins(DefaultPlugins)
        .add_plugin(EguiPlugin)
        .add_plugin(AnimatorControllerPlugin)
        .run();
}
