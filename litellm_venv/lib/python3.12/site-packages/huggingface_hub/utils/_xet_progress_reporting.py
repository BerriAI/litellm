from collections import OrderedDict
from typing import List

from hf_xet import PyItemProgressUpdate, PyTotalProgressUpdate

from .tqdm import tqdm


class XetProgressReporter:
    def __init__(self, n_lines: int = 10, description_width: int = 40):
        self.n_lines = n_lines
        self.description_width = description_width

        self.tqdm_settings = {
            "unit": "B",
            "unit_scale": True,
            "leave": True,
            "unit_divisor": 1000,
            "nrows": n_lines + 3,
            "miniters": 1,
            "bar_format": "{l_bar}{bar}| {n_fmt:>5}B / {total_fmt:>5}B{postfix:>12}",
        }

        # Overall progress bars
        self.data_processing_bar = tqdm(
            total=0,
            desc=self.format_desc("Processing Files (0 / 0)", False),
            position=0,
            **self.tqdm_settings,
        )

        self.upload_bar = tqdm(
            total=0,
            desc=self.format_desc("New Data Upload", False),
            position=1,
            **self.tqdm_settings,
        )

        self.known_items: set[str] = set()
        self.completed_items: set[str] = set()

        # Item bars (scrolling view)
        self.item_state: OrderedDict[str, PyItemProgressUpdate] = OrderedDict()
        self.current_bars: List = [None] * self.n_lines

    def format_desc(self, name: str, indent: bool) -> str:
        """
        if name is longer than width characters, prints ... at the start and then the last width-3 characters of the name, otherwise
        the whole name right justified into 20 characters.  Also adds some padding.
        """
        padding = "  " if indent else ""
        width = self.description_width - len(padding)

        if len(name) > width:
            name = f"...{name[-(width - 3) :]}"

        return f"{padding}{name.ljust(width)}"

    def update_progress(
        self,
        total_update: PyTotalProgressUpdate,
        item_updates: List[PyItemProgressUpdate],
    ):
        # Update all the per-item values.
        for item in item_updates:
            item_name = item.item_name

            self.known_items.add(item_name)

            # Only care about items where the processing has already started.
            if item.bytes_completed == 0:
                continue

            # Overwrite the existing value in there.
            self.item_state[item_name] = item

        bar_idx = 0
        new_completed = []

        # Now, go through and update all the bars
        for name, item in self.item_state.items():
            # Is this ready to be removed on the next update?
            if item.bytes_completed == item.total_bytes:
                self.completed_items.add(name)
                new_completed.append(name)

            # If we've run out of bars to use, then collapse the last ones together.
            if bar_idx >= len(self.current_bars):
                bar = self.current_bars[-1]
                in_final_bar_mode = True
                final_bar_aggregation_count = bar_idx + 1 - len(self.current_bars)
            else:
                bar = self.current_bars[bar_idx]
                in_final_bar_mode = False

            if bar is None:
                self.current_bars[bar_idx] = tqdm(
                    desc=self.format_desc(name, True),
                    position=2 + bar_idx,  # Set to the position past the initial bars.
                    total=item.total_bytes,
                    initial=item.bytes_completed,
                    **self.tqdm_settings,
                )

            elif in_final_bar_mode:
                bar.n += item.bytes_completed
                bar.total += item.total_bytes
                bar.set_description(
                    self.format_desc(f"[+ {final_bar_aggregation_count} files]", True),
                    refresh=False,
                )
            else:
                bar.set_description(self.format_desc(name, True), refresh=False)
                bar.n = item.bytes_completed
                bar.total = item.total_bytes

            bar_idx += 1

        # Remove all the completed ones from the ordered dictionary
        for name in new_completed:
            # Only remove ones from consideration to make room for more items coming in.
            if len(self.item_state) <= self.n_lines:
                break

            del self.item_state[name]

        # Now manually refresh each of the bars
        for bar in self.current_bars:
            if bar:
                bar.refresh()

        # Update overall bars
        def postfix(speed):
            s = tqdm.format_sizeof(speed) if speed is not None else "???"
            return f"{s}B/s  ".rjust(10, " ")

        self.data_processing_bar.total = total_update.total_bytes
        self.data_processing_bar.set_description(
            self.format_desc(
                f"Processing Files ({len(self.completed_items)} / {len(self.known_items)})",
                False,
            ),
            refresh=False,
        )
        self.data_processing_bar.set_postfix_str(
            postfix(total_update.total_bytes_completion_rate), refresh=False
        )
        self.data_processing_bar.update(total_update.total_bytes_completion_increment)

        self.upload_bar.total = total_update.total_transfer_bytes
        self.upload_bar.set_postfix_str(
            postfix(total_update.total_transfer_bytes_completion_rate), refresh=False
        )
        self.upload_bar.update(total_update.total_transfer_bytes_completion_increment)

    def close(self, _success):
        self.data_processing_bar.close()
        self.upload_bar.close()
        for bar in self.current_bars:
            if bar:
                bar.close()
