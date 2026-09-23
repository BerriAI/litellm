#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RedisNode {
    pub host: String,
    pub port: u16,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum RedisTopology {
    #[default]
    Standalone,
    Cluster {
        startup_nodes: Vec<RedisNode>,
    },
}
