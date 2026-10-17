"""Real Tcl/Tk lifecycle checks; no ADB or user configuration is accessed."""
import gc
import os
import time
import tkinter as tk
import unittest
import weakref
from types import SimpleNamespace
from unittest.mock import Mock

import customtkinter as ctk

from autoclicker.gui.main_window import App
from autoclicker.gui.tab_view import InstanceTabFrame
from autoclicker.gui.widgets.tooltip import TemplatePreviewTooltip


class MinimalTab(InstanceTabFrame):
    def __init__(self, master):
        ctk.CTkFrame.__init__(self, master)
        self.primary_search_var = tk.StringVar(master=self)
        self.fallback_search_var = tk.StringVar(master=self)
        self._primary_search_trace = self.primary_search_var.trace_add(
            "write", lambda *args: self.winfo_exists())
        self._fallback_search_trace = self.fallback_search_var.trace_add(
            "write", lambda *args: self.winfo_exists())
        self._flash_restore_id = self.after(60000, lambda: self.winfo_exists())

    def begin_shutdown(self):
        pass

    def shutdown(self):
        pass


class MemoryLifecycleTests(unittest.TestCase):
    def setUp(self):
        if os.environ.get("CI") == "true":
            raise unittest.SkipTest("Skip real Tk window lifecycle test on headless CI")
        try:
            self.root = ctk.CTk()
            self.root.withdraw()
        except Exception as e:
            raise unittest.SkipTest(f"Tk display not available: {e}")

    def tearDown(self):
        TemplatePreviewTooltip.release()
        # These tests own the entire interpreter, including CTk's global
        # polling jobs; cancel them before creating another test root.
        try:
            for timer_id in self.root.tk.splitlist(self.root.tk.call("after", "info")):
                try:
                    self.root.tk.call("after", "cancel", timer_id)
                except tk.TclError:
                    pass
        except tk.TclError:
            pass
        self.root.destroy()
        gc.collect()

    def test_repeated_tab_removal_releases_widgets_traces_and_objects(self):
        tabs = ctk.CTkTabview(self.root)
        tabs.add("keep")
        app = SimpleNamespace(tabview=tabs, tab_frames={"keep": Mock()},
                              save_app_config=Mock(), _alerts={}, _retiring=set(), after=self.root.after)
        refs = []
        for index in range(30):
            name = f"tab-{index}"
            container = tabs.add(name)
            frame = MinimalTab(container)
            variable = frame.primary_search_var
            refs.append(weakref.ref(frame))
            app.tab_frames[name] = frame
            App.remove_instance_tab(app, name)
            deadline = time.monotonic() + 2
            while container.winfo_exists() and time.monotonic() < deadline:
                self.root.update()
                time.sleep(0.01)
            self.assertFalse(container.winfo_exists())
            self.assertFalse(variable.trace_info())
            del variable, frame, container
        self.root.update_idletasks()
        gc.collect()
        self.assertEqual(sum(ref() is not None for ref in refs), 0)
        self.assertEqual(len(app.tab_frames), 1)

    def test_log_tag_count_stays_bounded_after_text_deletion(self):
        textbox = tk.Text(self.root)
        frame = SimpleNamespace(log_textbox=SimpleNamespace(_textbox=textbox))
        for index in range(500):
            textbox.insert("end", "template.png\n")
            textbox.tag_add(f"tpl_target::{index}.png", "1.0", "1.end")
            textbox.delete("1.0", "end")
            InstanceTabFrame._prune_log_tags(frame)
        self.assertFalse(any(tag.startswith("tpl_target::")
                             for tag in textbox.tag_names()))

    def test_tooltip_release_drops_root_and_cached_images(self):
        tooltip = TemplatePreviewTooltip.get_instance(self.root)
        widget = tk.Label(self.root)
        tooltip.schedule_show(widget, "missing.png")
        TemplatePreviewTooltip._image_cache["sample"] = tk.PhotoImage(
            master=self.root, width=10, height=10)
        TemplatePreviewTooltip._image_cache_order.append("sample")
        TemplatePreviewTooltip.release()
        self.assertIsNone(tooltip.root)
        self.assertIsNone(tooltip._scheduled_widget)
        self.assertIsNone(tooltip._hover_after_id)
        self.assertIsNone(TemplatePreviewTooltip._instance)
        self.assertFalse(TemplatePreviewTooltip._image_cache)
        self.assertFalse(TemplatePreviewTooltip._image_cache_order)


class FullAppLifecycleTests(unittest.TestCase):
    def setUp(self):
        if os.environ.get("CI") == "true":
            raise unittest.SkipTest("Skip full Tk app lifecycle test on headless CI")

    def test_five_tabs_alerts_and_async_close(self):
        import tempfile
        from unittest.mock import patch
        from autoclicker.core.clicker import AutoClicker
        with tempfile.TemporaryDirectory() as folder:
            with patch("autoclicker.gui.main_window.CONFIG_PATH", folder + "/config.json"), patch(
                    "autoclicker.gui.main_window.get_app_dir", return_value=folder), patch(
                    "autoclicker.gui.tab_view.AutoClicker",
                    side_effect=lambda **kwargs: AutoClicker(base_dir=folder, **kwargs)), patch.object(
                    App, "refresh_all_devices"), patch.object(App, "_preload_configured_templates"):
                app = App()
                callback_errors = []
                app.report_callback_exception = lambda *error: callback_errors.append(error)
                app.withdraw()
                for index in range(4):
                    app.add_instance_tab(f"emulator-{6000 + index * 2}")
                for frame in app.tab_frames.values():
                    frame.clicker.is_running = True
                    frame.on_no_match_timeout(150)
                deadline = time.monotonic() + 10
                while len(app._alerts) < 5 and time.monotonic() < deadline:
                    app.update()
                    time.sleep(0.01)
                self.assertEqual(len(app._alerts), 5)
                self.assertIsNone(app.grab_current())
                self.assertEqual(len(app._latest_updates), 0)
                app.on_closing()
                deadline = time.monotonic() + 10
                while not app._shutdown_done.is_set() and time.monotonic() < deadline:
                    app.update()
                    time.sleep(0.01)
                self.assertTrue(app._shutdown_done.is_set())
                # Drive the scheduled final destruction and release diagnostics.
                for _ in range(10):
                    try:
                        app.update()
                    except tk.TclError:
                        break
                    time.sleep(0.01)
                self.assertFalse(app._settings_writer.worker.is_alive())
                self.assertFalse(app._diagnostics.worker.is_alive())
                self.assertEqual(callback_errors, [])


if __name__ == "__main__":
    unittest.main()
