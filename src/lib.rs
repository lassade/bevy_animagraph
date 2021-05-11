mod nodes;
mod state;

pub use nodes::*;
pub use state::*;

pub struct AnimatorController {
    //pub node: Handle<Box<dyn Node + 'static>>,
    pub node: Box<dyn Node + 'static>,
    pub state: State,
}
