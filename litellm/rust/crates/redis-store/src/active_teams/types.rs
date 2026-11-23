const SEPARATOR: char = '\u{1f}';

/// One caller active on a model and the priority class it last sent with.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ActiveMember {
    pub member: String,
    pub priority_class: String,
}

impl ActiveMember {
    /// The sorted set member: the id, the separator, the class.
    pub(super) fn spelling(&self) -> String {
        format!("{}{SEPARATOR}{}", self.member, self.priority_class)
    }

    /// The class is split off from the right, since an id may hold the separator itself.
    pub(super) fn read(spelling: &str) -> Option<Self> {
        let (member, priority_class) = spelling.rsplit_once(SEPARATOR)?;
        Some(Self {
            member: member.to_owned(),
            priority_class: priority_class.to_owned(),
        })
    }
}
