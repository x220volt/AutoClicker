"""Container for widgets in a template row, maintaining backward-compatible indexing."""


class TemplateRowWidgets:
    """Container for widgets in a template row, maintaining backward-compatible indexing."""

    def __init__(
        self, frame, priority, drag, action_btn, delay_btn, count_label, label, del_btn
    ):
        self.frame = frame
        self.priority = priority
        self.drag = drag
        self.action_btn = action_btn
        self.delay_btn = delay_btn
        self.count_label = count_label
        self.label = label
        self.del_btn = del_btn

    def __getitem__(self, index):
        if index == 0:
            return self.frame
        elif index == 1:
            return self.priority
        elif index == 2:
            return self.drag
        raise IndexError(f"Index {index} out of range for TemplateRowWidgets")
