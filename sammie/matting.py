# sammie/matting.py
import cv2
import os
import json
import numpy as np
import torch
import gc
from tqdm import tqdm
from PySide6.QtWidgets import QProgressDialog, QApplication
from PySide6.QtCore import Qt
from sammie import core
from sammie.settings_manager import get_settings_manager
from sammie.model_downloader import ensure_models
from sammie.trimap import TrimapConfig, generate_trimap
from sammie.vitmatte_backend import VitMatteBackend, VitMatteUnavailableError
from sammie.mematte_backend import MematteBackend, MematteUnavailableError
from sammie.hybrid_hq import (
    PRESERVE_TEMPORAL,
    alpha_to_float,
    apply_edge_residual,
    build_edge_residual,
    get_hybrid_merge_preset,
    stabilize_edge_residual,
    temporal_alpha_to_trimap,
)
from sammie.motion_confidence import (
    calculate_bidirectional_alignment,
    motion_confidence_gate,
    warp_source_to_target,
)
from sammie.hybrid_evaluation import (
    HybridEvaluationCancelled,
    evaluate_hybrid_run,
)
from sammie.performance_metrics import MattingRunProfiler, write_performance_report


class MattingManager:
    """
    Shared base class for matting managers.
    Provides common infrastructure: callbacks, image resize/restore, mask loading,
    progress dialog helpers, and matting directory management.
    Subclasses must implement load_matting_model() and run_matting().
    """

    def __init__(self):
        self.processor = None
        self.propagated = False  # whether we have propagated the mattes
        self.callbacks = []

    def add_callback(self, callback):
        """Add callback for matting events"""
        self.callbacks.append(callback)

    def _notify(self, action, **kwargs):
        """Notify callbacks of changes"""
        for callback in self.callbacks:
            try:
                callback(action, **kwargs)
            except RuntimeError as e:
                # Allow cancellation to propagate
                if str(e) == "USER_CANCELLED":
                    raise
                print(f"Callback error: {e}")
            except Exception as e:
                print(f"Callback error: {e}")

    def _prepare_device(self, load_to_cpu=False):
        """Return the appropriate torch device and clear cache"""
        core.DeviceManager.clear_cache()
        if load_to_cpu:
            return torch.device('cpu')
        return core.DeviceManager.get_device()

    def unload_matting_model(self):
        """Unload the matting model and clear cache"""
        self.processor = None
        gc.collect()
        core.DeviceManager.clear_cache()
        print("Unloaded Matting model")

    def _resize_image(self, image):
            """Resize image and ensure dimensions are multiples of 8 for the model."""
            settings_mgr = get_settings_manager()
            max_size = settings_mgr.get_session_setting("matany_res", 0)
            h, w = image.shape[:2]
            
            # 1. Determine scaling factor
            scale = 1.0
            if max_size > 0:
                min_side = min(h, w)
                if min_side > max_size:
                    scale = max_size / min_side

            # 2. ALWAYS round to a multiple of 8
            # This ensures [3, 1384, 600] instead of [3, 1390, 602]
            new_h = (int(h * scale) // 8) * 8
            new_w = (int(w * scale) // 8) * 8
            
            # 3. Always resize, even if scale is 1.0, to catch those extra pixels
            return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    def _restore_image_size(self, image, original_size):
        """Restore image to original size. original_size must be (w, h) as expected by cv2."""
        original_w, original_h = original_size
        restored_image = cv2.resize(image, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
        return restored_image

    def _load_mask_for_matting(self, object_id, frame_number, device, combine_ids=None):
        """
        Load and validate a mask for matting processing.

        If combine_ids is provided, masks for all IDs in that list are unioned
        in memory and returned as a single mask, without touching any files on disk.
        object_id is used only as the label for error messages in that case.

        Args:
            object_id: ID of the object (or output label when combining)
            frame_number: Frame number
            device: Processing device
            combine_ids: Optional list of object IDs to union into a single mask

        Returns:
            tuple: (mask_tensor, original_size) or (None, None) if failed
        """
        if combine_ids:
            union_mask = None
            original_size = None
            for oid in combine_ids:
                mask_filename = os.path.join(core.mask_dir, f"{frame_number:05d}", f"{oid}.png")
                if not os.path.exists(mask_filename):
                    continue
                m = cv2.imread(mask_filename, cv2.IMREAD_GRAYSCALE)
                if m is None:
                    continue
                if original_size is None:
                    original_size = m.shape[1::-1]
                union_mask = m if union_mask is None else np.maximum(union_mask, m)
            if union_mask is None or not np.any(union_mask):
                print(f"Combined mask is blank or missing for frame {frame_number}")
                return None, None
            mask = self._resize_image(union_mask)
            mask = torch.tensor(mask, dtype=torch.float32, device=device)
            return mask, original_size

        mask_filename = os.path.join(core.mask_dir, f"{frame_number:05d}", f"{object_id}.png")
        if not os.path.exists(mask_filename):
            print(f"Mask not found for object {object_id} at frame {frame_number}: {mask_filename}")
            return None, None

        mask = cv2.imread(mask_filename, cv2.IMREAD_GRAYSCALE)
        if mask is None or not np.any(mask):
            print(f"Mask is blank or invalid for object {object_id} at frame {frame_number}")
            return None, None

        original_size = mask.shape[1::-1]
        mask = core.apply_mask_postprocessing(mask)
        mask = self._resize_image(mask)
        mask = torch.tensor(mask, dtype=torch.float32, device=device)

        return mask, original_size

    def _make_progress_dialog(self, parent_window, total_operations, unit="frame"):
        """Create and show the Qt progress dialog and tqdm bar"""
        progress_dialog = QProgressDialog("Running matting...", "Cancel", 0, 100, parent_window)
        progress_dialog.setWindowTitle("Matting Progress")
        progress_dialog.setWindowModality(Qt.WindowModal)
        progress_dialog.setAutoClose(True)
        progress_dialog.show()
        pbar = tqdm(total=total_operations, desc="Matting Progress", unit=unit)
        return progress_dialog, pbar

    def _get_frame_range(self):
        """
        Read in/out points from settings and return (start_frame, end_frame, frames_to_process).
        """
        settings_mgr = get_settings_manager()
        frame_count = core.VideoInfo.total_frames
        in_point = settings_mgr.get_session_setting("in_point", None)
        out_point = settings_mgr.get_session_setting("out_point", None)
        start_frame = in_point if in_point is not None else 0
        end_frame = out_point if out_point is not None else frame_count - 1
        frames_to_process = end_frame - start_frame + 1
        return start_frame, end_frame, frames_to_process

    def _collect_image_paths(self, start_frame, end_frame):
        """Return a list of existing frame image paths in [start_frame, end_frame]."""
        extension = core.get_frame_extension()
        images = []
        for frame_number in range(start_frame, end_frame + 1):
            image_filename = os.path.join(core.frames_dir, f"{frame_number:05d}.{extension}")
            if os.path.exists(image_filename):
                images.append(image_filename)
        return images

    def clear_matting(self):
        """Clear matting data"""
        import shutil
        if os.path.exists(core.matting_dir):
            shutil.rmtree(core.matting_dir)
        os.makedirs(core.matting_dir)
        self.propagated = False
        print("Matting data cleared")

    def load_matting_model(self, load_to_cpu=False, parent_window=None):
        raise NotImplementedError("Subclasses must implement load_matting_model()")

    def run_matting(self, points_list, parent_window, combined=False):
        raise NotImplementedError("Subclasses must implement run_matting()")


class ImageMattingManager(MattingManager):
    """Base contract for original-resolution per-frame matting backends."""

    def _trimap_config(self):
        settings_mgr = get_settings_manager()
        automatic = settings_mgr.get_session_setting("trimap_auto", True)
        if automatic:
            return TrimapConfig()
        return TrimapConfig(
            erode_width=settings_mgr.get_session_setting("trimap_fg_erode", 8),
            dilate_width=settings_mgr.get_session_setting("trimap_bg_dilate", 8),
        )

    def _load_source_mask(self, frame_number, object_ids):
        union = None
        for object_id in object_ids:
            filename = os.path.join(
                core.mask_dir, f"{frame_number:05d}", f"{object_id}.png"
            )
            mask = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            mask = core.apply_mask_postprocessing(mask)
            union = mask if union is None else np.maximum(union, mask)
        return union

    def _make_trimap(self, frame_number, output_id, object_ids):
        mask = self._load_source_mask(frame_number, object_ids)
        if mask is None or not np.any(mask):
            return None
        trimap = generate_trimap(mask, self._trimap_config())
        filename = os.path.join(
            core.trimap_dir, f"{frame_number:05d}", f"{output_id}.png"
        )
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        if not cv2.imwrite(filename, trimap):
            raise OSError(f"Failed to write trimap: {filename}")
        return trimap


# ---------------------------------------------------------------------------
# MatAnyone backend
# ---------------------------------------------------------------------------

class MatAnyManager(MattingManager):
    """Matting manager that uses the MatAnyone / MatAnyone2 model."""

    BACKEND = "matanyone"

    def __init__(self, model_name=None):
        super().__init__()
        self.model_name = model_name

    def load_matting_model(self, load_to_cpu=False, parent_window=None):
        """Load the MatAnyone model and return processor"""
        from matanyone.inference.inference_core import InferenceCore
        from matanyone.utils.get_default_model import get_matanyone_model

        device = self._prepare_device(load_to_cpu)
        settings_mgr = get_settings_manager()
        matting_model = self.model_name or settings_mgr.get_session_setting(
            "matany_model", "MatAnyone2"
        )
        max_size = settings_mgr.get_session_setting("matany_res", 0)
        combined = settings_mgr.get_session_setting("matany_combined", False)

        if matting_model == "MatAnyone2":
            checkpoint = "./checkpoints/matanyone2.pth"
            if not ensure_models("matanyone2", parent=parent_window):
                return False  # user cancelled or download failed
        else:
            checkpoint = "./checkpoints/matanyone.pth"
            if not ensure_models("matanyone", parent=parent_window):
                return False  # user cancelled or download failed

        matanyone = get_matanyone_model(checkpoint, device=device)
        print(f"Loaded {matting_model} model to {device} with max size {max_size} and combined={combined}")

        # Initialize inference processor
        self.processor = InferenceCore(matanyone, cfg=matanyone.cfg, device=device)
        return self.processor

    @torch.inference_mode()
    def run_matting(self, points_list, parent_window, combined=False):
        """
        Run matting on all frames, using multiple keyframes for each object.

        Args:
            points_list (list): List of point dictionaries containing object_id and frame information
            parent_window: Parent window for progress dialog

        Returns:
            int: 1 if successful, 0 if cancelled/failed
        """
        if self.processor is None:
            print("Matting model not loaded")
            return 0

        core.DeviceManager.clear_cache()
        device = core.DeviceManager.get_device()
        frame_count = core.VideoInfo.total_frames

        start_frame, end_frame, frames_to_process = self._get_frame_range()
        print(f"Processing matting from frame {start_frame} to {end_frame} ({frames_to_process} frames)")

        # Get unique object IDs from points list
        object_ids = sorted(list(set(point['object_id'] for point in points_list if 'object_id' in point)))
        if not object_ids:
            print("No objects found for matting")
            return 0

        # Find all keyframes for each object (within processing range)
        object_keyframes = {}
        for object_id in object_ids:
            keyframes = sorted(list(set(
                point['frame'] for point in points_list
                if point.get('object_id') == object_id and start_frame <= point['frame'] <= end_frame
            )))
            if keyframes:
                object_keyframes[object_id] = keyframes
            else:
                print(f"No frames found for object {object_id} in range {start_frame}-{end_frame}")

        if not object_keyframes:
            print("No valid keyframes found for any objects in the specified range")
            return 0

        # When combined mode is requested, run a single pass using the union of all
        # object masks loaded in memory — no files are written to disk.
        if combined and len(object_ids) > 1:
            combine_ids = object_ids
            earliest_keyframe = min(kf[0] for kf in object_keyframes.values())
            object_ids = [0]
            object_keyframes = {0: [earliest_keyframe]}
        else:
            combine_ids = None

        # Calculate total operations for progress tracking
        total_operations = 0
        for object_id, keyframes in object_keyframes.items():
            first_keyframe = keyframes[0]

            # Operations before first keyframe (backward propagation to start_frame)
            total_operations += first_keyframe - start_frame

            # Operations between keyframes and after last keyframe to end_frame
            for i in range(len(keyframes)):
                if i == len(keyframes) - 1:
                    total_operations += end_frame - keyframes[i] + 1
                else:
                    total_operations += keyframes[i + 1] - keyframes[i]

        progress_dialog, pbar = self._make_progress_dialog(parent_window, total_operations)

        # Create matting directory if it doesn't exist
        os.makedirs(core.matting_dir, exist_ok=True)

        # If combined mode is selected, delete any existing matting files except object 0.
        if combined and os.path.exists(core.matting_dir):
            for frame_dirname in os.listdir(core.matting_dir):
                frame_dir = os.path.join(core.matting_dir, frame_dirname)
                if os.path.isdir(frame_dir):
                    for f in os.listdir(frame_dir):
                        if f != "0.png":
                            os.remove(os.path.join(frame_dir, f))

        images = self._collect_image_paths(start_frame, end_frame)

        operations_completed = 0

        # Process each object with its keyframes
        for object_id, keyframes in object_keyframes.items():
            if progress_dialog.wasCanceled():
                break

            pbar.set_description(f"Object {object_id}")

            # Process segments for this object
            success = self._process_object_with_keyframes(
                images, object_id, keyframes, end_frame + 1, device,
                progress_dialog, operations_completed, total_operations, pbar, parent_window,
                start_frame=start_frame, combine_ids=combine_ids
            )

            if not success:
                break

            # Update operations completed for this object
            first_keyframe = keyframes[0]
            operations_completed += first_keyframe - start_frame  # backward from first keyframe

            for i in range(len(keyframes)):
                if i == len(keyframes) - 1:
                    operations_completed += end_frame - keyframes[i] + 1  # last keyframe to end
                else:
                    operations_completed += keyframes[i + 1] - keyframes[i]  # between keyframes

        # Close tqdm progress bar
        pbar.close()

        # Final cleanup
        core.DeviceManager.clear_cache()

        if progress_dialog.wasCanceled():
            print("Matting cancelled")
            self.propagated = False
            progress_dialog.close()
            return 0
        else:
            progress_dialog.setValue(100)
            if frame_count == frames_to_process:
                self.propagated = True  # only set propagated to True if the entire video was processed
            else:
                self.propagated = False
            print("Matting completed")
            self._notify('matting_complete')
            return 1

    def _process_object_with_keyframes(self, images, object_id, keyframes, frame_count, device, progress_dialog,
                                       operations_completed, total_operations, pbar, parent_window, start_frame=0,
                                       combine_ids=None):
        """
        Process a single object using multiple keyframes.

        Args:
            images: List of image paths
            object_id: ID of the object to process (also the output file label)
            keyframes: Sorted list of keyframe indices for this object
            frame_count: Total number of frames
            device: Processing device
            progress_dialog: Progress dialog for user feedback
            operations_completed: Number of operations completed so far
            total_operations: Total operations for all objects
            pbar: tqdm progress bar
            parent_window: Parent window
            start_frame: Starting frame for the processing range
            combine_ids: If set, union masks for these IDs in memory rather than
                         loading a single object mask from disk

        Returns:
            bool: True if successful, False if cancelled or failed
        """
        first_keyframe = keyframes[0]

        # Load and validate the first keyframe mask
        mask, original_size = self._load_mask_for_matting(object_id, first_keyframe, device,
                                                          combine_ids=combine_ids)
        if mask is None:
            return False

        # Special case for single frame
        if len(images) == 1:
            return self._process_single_frame(images[0], mask, object_id, original_size, device)

        current_operations = operations_completed

        # 1. Process backward from first keyframe to start_frame
        if first_keyframe > start_frame:
            success = self._process_backward(images, mask, object_id, first_keyframe,
                                             original_size, device, progress_dialog, current_operations,
                                             total_operations, parent_window, pbar, start_frame_offset=start_frame)
            if not success:
                return False
            current_operations += first_keyframe - start_frame

        # 2. Process forward segments between keyframes
        for i in range(len(keyframes)):
            if progress_dialog.wasCanceled():
                return False

            current_keyframe = keyframes[i]

            # Load mask for current keyframe (refresh for each segment)
            mask, original_size = self._load_mask_for_matting(object_id, current_keyframe, device,
                                                              combine_ids=combine_ids)
            if mask is None:
                print(f"Failed to load mask for object {object_id} at keyframe {current_keyframe}")
                return False

            # Determine end frame for this segment
            if i == len(keyframes) - 1:
                end_frame = frame_count  # Last keyframe - process to end of range
            else:
                end_frame = keyframes[i + 1]  # Process to next keyframe (exclusive)

            # Process this forward segment
            if end_frame > current_keyframe:
                success = self._process_forward(images, mask, object_id, current_keyframe, original_size,
                                                device, progress_dialog, current_operations, total_operations,
                                                parent_window, end_frame, pbar, start_frame_offset=start_frame)
                if not success:
                    return False
                current_operations += end_frame - current_keyframe

        return True

    def _process_single_frame(self, frame_path, mask, object_id, original_size, device):
        """Process a single frame for matting"""
        try:
            img = cv2.imread(frame_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = self._resize_image(img)
            img = torch.tensor(img / 255., dtype=torch.float32, device=device).permute(2, 0, 1)

            output_prob = self.processor.step(img, mask, objects=[1])
            for i in range(10):  # Warmup iterations
                output_prob = self.processor.step(img, first_frame_pred=True)
                core.DeviceManager.clear_cache()

            mat = self.processor.output_prob_to_mask(output_prob)
            mat = mat.detach().cpu().numpy()
            mat = (mat * 255).astype(np.uint8)
            mat = self._restore_image_size(mat, original_size)

            mat_filename = os.path.join(core.matting_dir, f"00000", f"{object_id}.png")
            os.makedirs(os.path.dirname(mat_filename), exist_ok=True)
            cv2.imwrite(mat_filename, mat)
            return True

        except Exception as e:
            print(f"Error processing single frame: {e}")
            return False

    def _process_forward(self, images, mask, object_id, start_frame, original_size, device, progress_dialog,
                         operations_completed, total_operations, parent_window, end_frame=None, pbar=None,
                         start_frame_offset=0):
        """
        Process frames forward from start_frame.

        Args:
            images: List of image paths
            mask: Initial mask tensor
            object_id: Object ID
            start_frame: Starting frame (inclusive)
            original_size: Original image size
            device: Processing device
            progress_dialog: Progress dialog
            operations_completed: Operations completed before this segment
            total_operations: Total operations
            parent_window: Parent window
            end_frame: Ending frame (exclusive). If None, process to end of images.
            pbar: tqdm progress bar
            start_frame_offset: Offset for mapping array indices to absolute frame numbers

        Returns:
            bool: True if successful, False if cancelled or failed
        """
        if end_frame is None:
            end_frame = start_frame + len(images)

        # Get display update frequency from settings
        settings_mgr = get_settings_manager()
        display_update_frequency = settings_mgr.get_app_setting("display_update_frequency", 5)

        try:
            for frame_number in range(start_frame, end_frame):
                if progress_dialog.wasCanceled():
                    return False

                # Map absolute frame number to array index
                array_idx = frame_number - start_frame_offset
                if array_idx < 0 or array_idx >= len(images):
                    print(f"Warning: Frame {frame_number} out of range for images array")
                    continue

                frame_path = images[array_idx]
                img = cv2.imread(frame_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = self._resize_image(img)
                img = torch.tensor(img / 255., dtype=torch.float32, device=device).permute(2, 0, 1)

                if frame_number == start_frame:
                    # First frame - initialize with mask
                    output_prob = self.processor.step(img, mask, objects=[1])
                    for i in range(10):  # Warmup iterations
                        output_prob = self.processor.step(img, first_frame_pred=True)
                        core.DeviceManager.clear_cache()
                else:
                    # Subsequent frames - propagate
                    output_prob = self.processor.step(img)

                # Convert to matte
                mat = self.processor.output_prob_to_mask(output_prob)
                mat = mat.detach().cpu().numpy()
                mat = (mat * 255).astype(np.uint8)
                mat = self._restore_image_size(mat, original_size)

                # Save matte
                mat_filename = os.path.join(core.matting_dir, f"{frame_number:05d}", f"{object_id}.png")
                os.makedirs(os.path.dirname(mat_filename), exist_ok=True)
                cv2.imwrite(mat_filename, mat)
                core.DeviceManager.clear_cache()

                # Update display at the specified frequency
                if frame_number % display_update_frequency == 0:
                    try:
                        parent_window.frame_slider.setValue(frame_number)
                    except Exception as e:
                        print(f"Error updating display: {e}")

                # Update progress
                if pbar is not None:
                    pbar.update(1)
                current_progress = int(((operations_completed + (frame_number - start_frame) + 1) * 100) / total_operations)
                progress_dialog.setValue(current_progress)
                QApplication.processEvents()

            return True

        except Exception as e:
            print(f"Error in forward processing: {e}")
            return False

    def _process_backward(self, images, mask, object_id, start_frame, original_size, device, progress_dialog,
                          operations_completed, total_operations, parent_window, pbar=None, start_frame_offset=0):
        """Process frames backward from start_frame"""

        # Get display update frequency from settings
        settings_mgr = get_settings_manager()
        display_update_frequency = settings_mgr.get_app_setting("display_update_frequency", 5)

        try:
            for frame_number in range(start_frame, start_frame_offset - 1, -1):
                if progress_dialog.wasCanceled():
                    return False

                # Map absolute frame number to array index
                array_idx = frame_number - start_frame_offset
                if array_idx < 0 or array_idx >= len(images):
                    print(f"Warning: Frame {frame_number} out of range for images array")
                    continue

                frame_path = images[array_idx]
                img = cv2.imread(frame_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = self._resize_image(img)
                img = torch.tensor(img / 255., dtype=torch.float32, device=device).permute(2, 0, 1)

                if frame_number == start_frame:
                    # First frame - initialize with mask
                    output_prob = self.processor.step(img, mask, objects=[1])
                    for i in range(10):  # Warmup iterations
                        output_prob = self.processor.step(img, first_frame_pred=True)
                        core.DeviceManager.clear_cache()
                else:
                    # Subsequent frames - propagate
                    output_prob = self.processor.step(img)

                # Convert to matte
                mat = self.processor.output_prob_to_mask(output_prob)
                mat = mat.detach().cpu().numpy()
                mat = (mat * 255).astype(np.uint8)
                mat = self._restore_image_size(mat, original_size)

                # Save matte
                mat_filename = os.path.join(core.matting_dir, f"{frame_number:05d}", f"{object_id}.png")
                os.makedirs(os.path.dirname(mat_filename), exist_ok=True)
                cv2.imwrite(mat_filename, mat)
                core.DeviceManager.clear_cache()

                # Update display at the specified frequency
                if frame_number % display_update_frequency == 0:
                    try:
                        parent_window.frame_slider.setValue(frame_number)
                    except Exception as e:
                        print(f"Error updating display: {e}")

                # Update progress
                if pbar is not None:
                    pbar.update(1)
                operations_completed += 1
                progress_dialog.setValue(operations_completed * 100 // total_operations)
                QApplication.processEvents()

            return True

        except Exception as e:
            print(f"Error in backward processing: {e}")
            return False


# ---------------------------------------------------------------------------
# ViTMatte image backend
# ---------------------------------------------------------------------------

class VitMatteManager(ImageMattingManager):
    """Original-resolution, ROI-based ViTMatte image matting manager."""

    BACKEND = "ViTMatte"

    def __init__(self):
        super().__init__()
        self.backend = None

    def load_matting_model(self, load_to_cpu=False, parent_window=None):
        device = self._prepare_device(load_to_cpu)
        self.backend = VitMatteBackend(device)
        try:
            self.backend.load()
        except VitMatteUnavailableError as exc:
            print(str(exc))
            self.backend = None
            return None
        self.processor = self.backend
        print(f"Loaded ViTMatte model to {device}")
        return self.processor

    def unload_matting_model(self):
        if self.backend is not None:
            self.backend.unload()
        self.backend = None
        self.processor = None
        gc.collect()
        core.DeviceManager.clear_cache()
        print("Unloaded ViTMatte model")

    def run_matting(self, points_list, parent_window, combined=False):
        if self.backend is None:
            print("ViTMatte model not loaded")
            return 0

        settings_mgr = get_settings_manager()
        start_frame, end_frame, frames_to_process = self._get_frame_range()
        source_object_ids = sorted({int(point["object_id"]) for point in points_list})
        if not source_object_ids:
            return 0
        jobs = [(0, source_object_ids)] if combined else [
            (object_id, [object_id]) for object_id in source_object_ids
        ]
        total_operations = frames_to_process * len(jobs)
        progress_dialog, pbar = self._make_progress_dialog(parent_window, total_operations)
        margin = settings_mgr.get_session_setting("vitmatte_roi_margin", 64)
        tile_size = settings_mgr.get_session_setting("vitmatte_tile_size", 1024)
        tile_overlap = settings_mgr.get_session_setting("vitmatte_tile_overlap", 128)
        display_frequency = settings_mgr.get_app_setting("display_update_frequency", 5)
        extension = core.get_frame_extension()
        completed = 0
        cancelled = False

        os.makedirs(core.matting_dir, exist_ok=True)
        if combined:
            for frame_dirname in os.listdir(core.matting_dir):
                frame_dir = os.path.join(core.matting_dir, frame_dirname)
                if os.path.isdir(frame_dir):
                    for filename in os.listdir(frame_dir):
                        if filename != "0.png":
                            os.remove(os.path.join(frame_dir, filename))

        try:
            for output_id, input_ids in jobs:
                pbar.set_description(f"ViTMatte object {output_id}")
                for frame_number in range(start_frame, end_frame + 1):
                    if progress_dialog.wasCanceled():
                        cancelled = True
                        break
                    frame_path = os.path.join(
                        core.frames_dir, f"{frame_number:05d}.{extension}"
                    )
                    bgr = cv2.imread(frame_path, cv2.IMREAD_COLOR)
                    if bgr is None:
                        raise OSError(f"Unable to read frame: {frame_path}")
                    trimap = self._make_trimap(frame_number, output_id, input_ids)
                    if trimap is None:
                        continue
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    alpha = self.backend.predict_multi_roi_float(
                        rgb,
                        trimap,
                        margin=margin,
                        max_tile_size=tile_size,
                        tile_overlap=tile_overlap,
                    )
                    output_path = os.path.join(
                        core.matting_dir, f"{frame_number:05d}", f"{output_id}.png"
                    )
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    alpha16 = np.round(np.clip(alpha, 0.0, 1.0) * 65535.0).astype(np.uint16)
                    if not cv2.imwrite(output_path, alpha16):
                        raise OSError(f"Failed to write matte: {output_path}")

                    completed += 1
                    pbar.update(1)
                    progress_dialog.setValue(int(completed * 100 / max(total_operations, 1)))
                    if frame_number % display_frequency == 0:
                        parent_window.frame_slider.setValue(frame_number)
                    QApplication.processEvents()
                if cancelled:
                    break
        finally:
            pbar.close()
            progress_dialog.close()
            core.DeviceManager.clear_cache()

        self.propagated = (
            not cancelled
            and start_frame == 0
            and end_frame == core.VideoInfo.total_frames - 1
        )
        if cancelled:
            print("ViTMatte matting cancelled")
            return 0
        print("ViTMatte matting completed")
        self._notify('matting_complete')
        return 1


# ---------------------------------------------------------------------------
# MEMatte image backend
# ---------------------------------------------------------------------------

class MematteManager(ImageMattingManager):
    """High-resolution, token-limited MEMatte image matting manager."""

    BACKEND = "MEMatte"

    def __init__(self):
        super().__init__()
        self.backend = None

    def load_matting_model(self, load_to_cpu=False, parent_window=None):
        settings_mgr = get_settings_manager()
        device = self._prepare_device(load_to_cpu)
        self.backend = MematteBackend(
            device,
            max_number_token=settings_mgr.get_session_setting(
                "mematte_max_tokens", 12000
            ),
            precision=settings_mgr.get_session_setting(
                "mematte_precision", "Float16"
            ),
        )
        try:
            self.backend.load()
        except MematteUnavailableError as exc:
            print(str(exc))
            self.backend = None
            return None
        self.processor = self.backend
        print(
            f"Loaded MEMatte model to {device} with max tokens "
            f"{self.backend.max_number_token} ({self.backend.precision})"
        )
        return self.processor

    def unload_matting_model(self):
        if self.backend is not None:
            self.backend.unload()
        self.backend = None
        self.processor = None
        gc.collect()
        core.DeviceManager.clear_cache()
        print("Unloaded MEMatte model")

    def run_matting(self, points_list, parent_window, combined=False):
        if self.backend is None:
            print("MEMatte model not loaded")
            return 0

        settings_mgr = get_settings_manager()
        start_frame, end_frame, frames_to_process = self._get_frame_range()
        source_object_ids = sorted({int(point["object_id"]) for point in points_list})
        if not source_object_ids:
            return 0
        jobs = [(0, source_object_ids)] if combined else [
            (object_id, [object_id]) for object_id in source_object_ids
        ]
        total_operations = frames_to_process * len(jobs)
        progress_dialog, pbar = self._make_progress_dialog(parent_window, total_operations)
        margin = settings_mgr.get_session_setting("mematte_roi_margin", 96)
        tile_size = settings_mgr.get_session_setting("mematte_tile_size", 2048)
        tile_overlap = settings_mgr.get_session_setting("mematte_tile_overlap", 128)
        self.backend.max_number_token = settings_mgr.get_session_setting(
            "mematte_max_tokens", 12000
        )
        self.backend.precision = settings_mgr.get_session_setting(
            "mematte_precision", "Float16"
        )
        display_frequency = settings_mgr.get_app_setting("display_update_frequency", 5)
        extension = core.get_frame_extension()
        completed = 0
        cancelled = False

        os.makedirs(core.matting_dir, exist_ok=True)
        if combined:
            for frame_dirname in os.listdir(core.matting_dir):
                frame_dir = os.path.join(core.matting_dir, frame_dirname)
                if os.path.isdir(frame_dir):
                    for filename in os.listdir(frame_dir):
                        if filename != "0.png":
                            os.remove(os.path.join(frame_dir, filename))

        try:
            for output_id, input_ids in jobs:
                pbar.set_description(f"MEMatte object {output_id}")
                for frame_number in range(start_frame, end_frame + 1):
                    if progress_dialog.wasCanceled():
                        cancelled = True
                        break
                    frame_path = os.path.join(
                        core.frames_dir, f"{frame_number:05d}.{extension}"
                    )
                    bgr = cv2.imread(frame_path, cv2.IMREAD_COLOR)
                    if bgr is None:
                        raise OSError(f"Unable to read frame: {frame_path}")
                    trimap = self._make_trimap(frame_number, output_id, input_ids)
                    if trimap is None:
                        continue
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    alpha = self.backend.predict_multi_roi_float(
                        rgb,
                        trimap,
                        margin=margin,
                        max_tile_size=tile_size,
                        tile_overlap=tile_overlap,
                    )
                    output_path = os.path.join(
                        core.matting_dir, f"{frame_number:05d}", f"{output_id}.png"
                    )
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    alpha16 = np.round(np.clip(alpha, 0.0, 1.0) * 65535.0).astype(
                        np.uint16
                    )
                    if not cv2.imwrite(output_path, alpha16):
                        raise OSError(f"Failed to write matte: {output_path}")

                    completed += 1
                    pbar.update(1)
                    progress_dialog.setValue(
                        int(completed * 100 / max(total_operations, 1))
                    )
                    if frame_number % display_frequency == 0:
                        parent_window.frame_slider.setValue(frame_number)
                    QApplication.processEvents()
                if cancelled:
                    break
        finally:
            pbar.close()
            progress_dialog.close()
            core.DeviceManager.clear_cache()

        self.propagated = (
            not cancelled
            and start_frame == 0
            and end_frame == core.VideoInfo.total_frames - 1
        )
        if cancelled:
            print("MEMatte matting cancelled")
            return 0
        print("MEMatte matting completed")
        self._notify('matting_complete')
        return 1


# ---------------------------------------------------------------------------
# Hybrid HQ staged temporal + spatial backend
# ---------------------------------------------------------------------------

class HybridHQManager(MattingManager):
    """Temporal matte followed by MEMatte refinement only in uncertain edges."""

    BACKEND = "Hybrid HQ"

    def __init__(self):
        super().__init__()
        self.temporal_manager = None
        self.edge_backend = None

    def load_matting_model(self, load_to_cpu=False, parent_window=None):
        # Large models are deliberately loaded inside run_matting one stage at a
        # time. Returning this lightweight coordinator satisfies the existing UI
        # lifecycle without co-resident temporal and spatial models.
        self.processor = self
        return self

    def unload_matting_model(self):
        if self.temporal_manager is not None:
            self.temporal_manager.unload_matting_model()
            self.temporal_manager = None
        if self.edge_backend is not None:
            self.edge_backend.unload()
            self.edge_backend = None
        self.processor = None
        gc.collect()
        core.DeviceManager.clear_cache()
        print("Unloaded Hybrid HQ stages")

    @staticmethod
    def _active_output_ids(points_list, start_frame, end_frame, combined, temporal):
        object_ids = sorted(
            {int(point["object_id"]) for point in points_list if "object_id" in point}
        )
        if temporal == "MatAnyone2":
            object_ids = [
                object_id
                for object_id in object_ids
                if any(
                    int(point.get("object_id", -1)) == object_id
                    and start_frame <= int(point.get("frame", -1)) <= end_frame
                    for point in points_list
                )
            ]
        if combined and len(object_ids) > 1:
            return [0]
        return object_ids

    def _create_temporal_manager(self, temporal_model):
        if temporal_model == "VideoMaMa":
            return VideoMaMaManager()
        return MatAnyManager(model_name="MatAnyone2")

    @staticmethod
    def _write_edge_proposal(
        proposal,
        preset,
        previous_residual=None,
        next_residual=None,
        current_residual=None,
        motion_confidence=None,
    ):
        residual = proposal["residual"] if current_residual is None else current_residual
        stable_residual = stabilize_edge_residual(
            residual,
            proposal["trimap"],
            preset=preset,
            previous_residual=previous_residual,
            next_residual=next_residual,
        )
        merged = apply_edge_residual(proposal["temporal"], stable_residual)
        merged16 = np.round(merged * 65535.0).astype(np.uint16)
        if not cv2.imwrite(proposal["matte_path"], merged16):
            raise OSError(
                f"Failed to write Hybrid HQ matte: {proposal['matte_path']}"
            )
        confidence_path = proposal.get("confidence_path")
        if motion_confidence is not None and confidence_path:
            os.makedirs(os.path.dirname(confidence_path), exist_ok=True)
            confidence8 = np.round(
                np.clip(motion_confidence, 0.0, 1.0) * 255.0
            ).astype(np.uint8)
            if not cv2.imwrite(confidence_path, confidence8):
                raise OSError(
                    f"Failed to write Hybrid motion confidence: {confidence_path}"
                )
        proposal["written"] = True
        unknown = proposal["trimap"] == 128
        if not unknown.any():
            proposal["temporal"] = None
            proposal["trimap"] = None
            return 0.0, 0, 0.0
        magnitude = np.abs(stable_residual[unknown])
        values = (
            float(magnitude.sum()),
            int(magnitude.size),
            float(magnitude.max()),
        )
        # A written proposal remains in the 3-frame buffer only as a residual
        # neighbor. Release its full alpha and trimap to keep 4K CPU RAM bounded.
        proposal["temporal"] = None
        proposal["trimap"] = None
        return values

    def _run_edge_stage(self, points_list, parent_window, combined, temporal_model):
        settings_mgr = get_settings_manager()
        start_frame, end_frame, frames_to_process = self._get_frame_range()
        output_ids = self._active_output_ids(
            points_list, start_frame, end_frame, combined, temporal_model
        )
        if not output_ids:
            print("Hybrid HQ found no temporal matte outputs to refine")
            return 0

        device = core.DeviceManager.get_device()
        self.edge_backend = MematteBackend(
            device,
            max_number_token=settings_mgr.get_session_setting(
                "mematte_max_tokens", 12000
            ),
            precision=settings_mgr.get_session_setting(
                "mematte_precision", "Float16"
            ),
        )
        try:
            self.edge_backend.load()
        except MematteUnavailableError as exc:
            print(str(exc))
            self.edge_backend = None
            return 0

        edge_width = settings_mgr.get_session_setting("hybrid_edge_width", 12)
        feather_width = settings_mgr.get_session_setting("hybrid_edge_feather", 4)
        merge_preset = settings_mgr.get_session_setting(
            "hybrid_stability_preset", PRESERVE_TEMPORAL
        )
        merge_config = get_hybrid_merge_preset(merge_preset)
        motion_enabled = settings_mgr.get_session_setting(
            "hybrid_motion_enabled", False
        )
        flow_resolution = settings_mgr.get_session_setting(
            "hybrid_flow_resolution", 720
        )
        margin = settings_mgr.get_session_setting("mematte_roi_margin", 96)
        tile_size = settings_mgr.get_session_setting("mematte_tile_size", 2048)
        tile_overlap = settings_mgr.get_session_setting("mematte_tile_overlap", 128)
        display_frequency = settings_mgr.get_app_setting("display_update_frequency", 5)
        extension = core.get_frame_extension()
        total_operations = frames_to_process * len(output_ids)
        progress_dialog, pbar = self._make_progress_dialog(
            parent_window, total_operations
        )
        progress_dialog.setWindowTitle("Hybrid HQ Edge Refinement")
        temporal_dir = os.path.join(core.temp_dir, "hybrid_temporal")
        trimap_dir = os.path.join(core.temp_dir, "hybrid_trimaps")
        confidence_dir = os.path.join(core.temp_dir, "hybrid_motion_confidence")
        completed = 0
        cancelled = False
        diagnostic_objects = []
        print(
            f"Hybrid HQ stability: {merge_config.name} "
            f"(residual limit={merge_config.residual_limit:.2f}, "
            f"temporal smoothing={merge_config.temporal_smoothing:.2f})"
        )
        if motion_enabled:
            print(
                "Hybrid HQ motion confidence: Experimental DIS "
                f"(flow resolution={flow_resolution})"
            )

        try:
            for object_id in output_ids:
                pbar.set_description(f"Hybrid HQ object {object_id}")
                proposal_buffer = []
                residual_sum = 0.0
                residual_count = 0
                residual_max = 0.0
                confidence_sum = 0.0
                confidence_count = 0
                for frame_number in range(start_frame, end_frame + 1):
                    if progress_dialog.wasCanceled():
                        cancelled = True
                        break

                    matte_path = os.path.join(
                        core.matting_dir, f"{frame_number:05d}", f"{object_id}.png"
                    )
                    temporal_raw = cv2.imread(matte_path, cv2.IMREAD_UNCHANGED)
                    if temporal_raw is None:
                        raise OSError(f"Hybrid HQ temporal matte is missing: {matte_path}")
                    temporal_alpha = alpha_to_float(temporal_raw)
                    trimap = temporal_alpha_to_trimap(
                        temporal_alpha, edge_width=edge_width
                    )

                    temporal_path = os.path.join(
                        temporal_dir, f"{frame_number:05d}", f"{object_id}.png"
                    )
                    trimap_path = os.path.join(
                        trimap_dir, f"{frame_number:05d}", f"{object_id}.png"
                    )
                    os.makedirs(os.path.dirname(temporal_path), exist_ok=True)
                    os.makedirs(os.path.dirname(trimap_path), exist_ok=True)
                    temporal16 = np.round(temporal_alpha * 65535.0).astype(np.uint16)
                    if not cv2.imwrite(temporal_path, temporal16):
                        raise OSError(f"Failed to save Hybrid HQ temporal matte: {temporal_path}")
                    if not cv2.imwrite(trimap_path, trimap):
                        raise OSError(f"Failed to save Hybrid HQ trimap: {trimap_path}")

                    bgr = None
                    if np.any(trimap == 128) or motion_enabled:
                        frame_path = os.path.join(
                            core.frames_dir, f"{frame_number:05d}.{extension}"
                        )
                        bgr = cv2.imread(frame_path, cv2.IMREAD_COLOR)
                        if bgr is None:
                            raise OSError(f"Unable to read frame: {frame_path}")
                    if np.any(trimap == 128):
                        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                        spatial_alpha = self.edge_backend.predict_multi_roi_float(
                            rgb,
                            trimap,
                            margin=margin,
                            max_tile_size=tile_size,
                            tile_overlap=tile_overlap,
                        )
                        residual = build_edge_residual(
                            temporal_alpha,
                            spatial_alpha,
                            trimap,
                            feather_width=feather_width,
                            preset=merge_config.name,
                        )
                    else:
                        residual = np.zeros_like(temporal_alpha, dtype=np.float32)

                    proposal = {
                        "frame_number": frame_number,
                        "matte_path": matte_path,
                        "temporal": temporal_alpha,
                        "trimap": trimap,
                        "residual": residual,
                        "gray": (
                            cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                            if motion_enabled else None
                        ),
                        "flow_from_previous": None,
                        "confidence_path": os.path.join(
                            confidence_dir,
                            f"{frame_number:05d}",
                            f"{object_id}.png",
                        ),
                        "written": False,
                    }
                    if motion_enabled and proposal_buffer:
                        try:
                            proposal["flow_from_previous"] = (
                                calculate_bidirectional_alignment(
                                    proposal_buffer[-1]["gray"],
                                    proposal["gray"],
                                    max_short_side=flow_resolution,
                                )
                            )
                        except (cv2.error, ValueError, RuntimeError) as exc:
                            print(
                                "Hybrid HQ motion alignment failed for frames "
                                f"{proposal_buffer[-1]['frame_number']}-"
                                f"{frame_number}; using Preserve Temporal "
                                f"fallback: {exc}"
                            )
                    proposal_buffer.append(proposal)

                    # The first frame has no previous neighbor, so duplicating its
                    # own residual makes the 3-frame median equal to itself.
                    if len(proposal_buffer) == 1:
                        values = self._write_edge_proposal(
                            proposal, merge_config.name
                        )
                        residual_sum += values[0]
                        residual_count += values[1]
                        residual_max = max(residual_max, values[2])
                    elif len(proposal_buffer) == 3:
                        previous, current, following = proposal_buffer
                        current_residual = current["residual"]
                        previous_residual = previous["residual"]
                        next_residual = following["residual"]
                        motion_confidence = None
                        if (
                            motion_enabled
                            and current["flow_from_previous"] is not None
                            and following["flow_from_previous"] is not None
                        ):
                            before = current["flow_from_previous"]
                            after = following["flow_from_previous"]
                            previous_warped, previous_confidence = (
                                warp_source_to_target(
                                    previous_residual,
                                    before.flow_current_to_previous,
                                    before.confidence_current,
                                    current_residual.shape,
                                )
                            )
                            next_warped, next_confidence = warp_source_to_target(
                                next_residual,
                                after.flow_previous_to_current,
                                after.confidence_previous,
                                current_residual.shape,
                            )
                            (
                                current_residual,
                                previous_residual,
                                next_residual,
                                motion_confidence,
                            ) = motion_confidence_gate(
                                current_residual,
                                previous_warped,
                                previous_confidence,
                                next_warped,
                                next_confidence,
                                agreement_sigma=merge_config.motion_agreement_sigma,
                                previous_fallback=previous["residual"],
                                next_fallback=following["residual"],
                            )
                            confidence_sum += float(motion_confidence.sum())
                            confidence_count += int(motion_confidence.size)
                        values = self._write_edge_proposal(
                            current,
                            merge_config.name,
                            previous_residual=previous_residual,
                            next_residual=next_residual,
                            current_residual=current_residual,
                            motion_confidence=motion_confidence,
                        )
                        residual_sum += values[0]
                        residual_count += values[1]
                        residual_max = max(residual_max, values[2])
                        proposal_buffer.pop(0)

                    completed += 1
                    pbar.update(1)
                    progress_dialog.setValue(
                        int(completed * 100 / max(total_operations, 1))
                    )
                    if frame_number % display_frequency == 0:
                        parent_window.frame_slider.setValue(frame_number)
                    QApplication.processEvents()
                if cancelled:
                    break

                if proposal_buffer:
                    last = proposal_buffer[-1]
                    if not last["written"]:
                        previous_residual = (
                            proposal_buffer[-2]["residual"]
                            if len(proposal_buffer) > 1
                            else None
                        )
                        values = self._write_edge_proposal(
                            last,
                            merge_config.name,
                            previous_residual=previous_residual,
                        )
                        residual_sum += values[0]
                        residual_count += values[1]
                        residual_max = max(residual_max, values[2])
                mean_residual = residual_sum / max(residual_count, 1)
                print(
                    f"Hybrid HQ object {object_id}: accepted residual "
                    f"mean={mean_residual:.6f}, max={residual_max:.6f}"
                )
                if motion_enabled:
                    mean_confidence = confidence_sum / max(confidence_count, 1)
                    print(
                        f"Hybrid HQ object {object_id}: motion confidence "
                        f"mean={mean_confidence:.6f}"
                    )
                else:
                    mean_confidence = None
                diagnostic_objects.append(
                    {
                        "object_id": int(object_id),
                        "accepted_residual_mean": float(mean_residual),
                        "accepted_residual_max": float(residual_max),
                        "motion_confidence_mean": (
                            float(mean_confidence)
                            if mean_confidence is not None else None
                        ),
                    }
                )
        finally:
            pbar.close()
            progress_dialog.close()
            self.edge_backend.unload()
            self.edge_backend = None
            gc.collect()
            core.DeviceManager.clear_cache()

        if cancelled:
            print("Hybrid HQ edge refinement cancelled")
            return 0
        diagnostic_dir = os.path.join(core.temp_dir, "hybrid_diagnostics")
        os.makedirs(diagnostic_dir, exist_ok=True)
        report_name = "_".join(
            (
                temporal_model.lower(),
                merge_config.name.lower().replace(" ", "_"),
                "motion" if motion_enabled else "phase41",
            )
        ) + ".json"
        report_path = os.path.join(diagnostic_dir, report_name)
        with open(report_path, "w", encoding="utf-8") as report_file:
            json.dump(
                {
                    "phase": "4.2" if motion_enabled else "4.1",
                    "temporal_model": temporal_model,
                    "stability_preset": merge_config.name,
                    "motion_enabled": bool(motion_enabled),
                    "flow_resolution": int(flow_resolution),
                    "frame_range": [int(start_frame), int(end_frame)],
                    "objects": diagnostic_objects,
                },
                report_file,
                indent=2,
            )
        print(f"Hybrid HQ diagnostics: {report_path}")
        return 1

    def _run_evaluation_stage(
        self, points_list, parent_window, combined, temporal_model
    ):
        settings_mgr = get_settings_manager()
        start_frame, end_frame, _ = self._get_frame_range()
        output_ids = self._active_output_ids(
            points_list, start_frame, end_frame, combined, temporal_model
        )
        if not output_ids:
            print("Hybrid HQ evaluation found no output objects")
            return None

        stability = settings_mgr.get_session_setting(
            "hybrid_stability_preset", PRESERVE_TEMPORAL
        )
        motion_enabled = settings_mgr.get_session_setting(
            "hybrid_motion_enabled", False
        ) is True
        flow_resolution = int(
            settings_mgr.get_session_setting("hybrid_flow_resolution", 720)
        )
        requested_label = settings_mgr.get_session_setting(
            "hybrid_evaluation_label", ""
        ).strip()
        run_label = requested_label or "_".join(
            (
                temporal_model.lower(),
                stability.lower().replace(" ", "_"),
                f"motion{flow_resolution}" if motion_enabled else "phase41",
            )
        )
        progress = QProgressDialog(
            "Preparing Hybrid HQ evaluation...", "Cancel", 0, 100, parent_window
        )
        progress.setWindowTitle("Hybrid HQ Phase 4.3 Evaluation")
        progress.setWindowModality(Qt.WindowModal)
        progress.setAutoClose(True)
        progress.show()

        def update(completed, total, message):
            progress.setLabelText(message)
            progress.setValue(int(completed * 100 / max(total, 1)))
            QApplication.processEvents()

        try:
            report_path = evaluate_hybrid_run(
                frames_dir=core.frames_dir,
                matting_dir=core.matting_dir,
                temporal_dir=os.path.join(core.temp_dir, "hybrid_temporal"),
                trimap_dir=os.path.join(core.temp_dir, "hybrid_trimaps"),
                confidence_dir=os.path.join(
                    core.temp_dir, "hybrid_motion_confidence"
                ),
                output_root=os.path.join(core.temp_dir, "hybrid_evaluation"),
                frame_range=(start_frame, end_frame),
                object_ids=output_ids,
                frame_extension=core.get_frame_extension(),
                run_label=run_label,
                settings={
                    "temporal_model": temporal_model,
                    "stability_preset": stability,
                    "motion_enabled": motion_enabled,
                    "flow_resolution": flow_resolution,
                    "edge_width": settings_mgr.get_session_setting(
                        "hybrid_edge_width", 12
                    ),
                    "edge_feather": settings_mgr.get_session_setting(
                        "hybrid_edge_feather", 4
                    ),
                },
                flow_resolution=flow_resolution,
                progress_callback=update,
                cancel_callback=progress.wasCanceled,
            )
            print(f"Hybrid HQ Phase 4.3 evaluation: {report_path}")
            return report_path
        except HybridEvaluationCancelled:
            print("Hybrid HQ Phase 4.3 evaluation cancelled; mattes were preserved")
        except Exception as exc:
            print(
                "Hybrid HQ Phase 4.3 evaluation failed; mattes were preserved: "
                f"{exc}"
            )
        finally:
            progress.close()
        return None

    def run_matting(self, points_list, parent_window, combined=False):
        settings_mgr = get_settings_manager()
        temporal_model = settings_mgr.get_session_setting(
            "hybrid_temporal_model", "MatAnyone2"
        )
        if temporal_model not in {"MatAnyone2", "VideoMaMa"}:
            raise ValueError(f"Unsupported Hybrid HQ temporal model: {temporal_model}")
        if core.DeviceManager.get_device().type == "cpu" and temporal_model == "VideoMaMa":
            print("Hybrid HQ with VideoMaMa is not supported on CPU")
            return 0

        device = core.DeviceManager.get_device()
        start_frame, end_frame, frames_to_process = self._get_frame_range()
        output_ids = self._active_output_ids(
            points_list, start_frame, end_frame, combined, temporal_model
        )
        frame_equivalents = frames_to_process * max(1, len(output_ids))
        memory_profile = settings_mgr.get_session_setting(
            "memory_profile", "Custom"
        )
        performance_enabled = settings_mgr.get_session_setting(
            "performance_metrics_enabled", True
        ) is True
        profiler = MattingRunProfiler(device)
        run_status = "failed"
        evaluation_enabled = settings_mgr.get_session_setting(
            "hybrid_evaluation_enabled", False
        ) is True
        stage_count = 3 if evaluation_enabled else 2
        try:
            print(f"Hybrid HQ stage 1/{stage_count}: {temporal_model} temporal matte")
            with profiler.stage("temporal", frame_equivalents):
                self.temporal_manager = self._create_temporal_manager(temporal_model)
                try:
                    if not self.temporal_manager.load_matting_model(
                        parent_window=parent_window
                    ):
                        run_status = "temporal_load_failed"
                        return 0
                    temporal_result = self.temporal_manager.run_matting(
                        points_list, parent_window=parent_window, combined=combined
                    )
                    temporal_propagated = self.temporal_manager.propagated
                finally:
                    if self.temporal_manager is not None:
                        self.temporal_manager.unload_matting_model()
                    self.temporal_manager = None
                    gc.collect()
                    core.DeviceManager.clear_cache()

            if temporal_result != 1:
                run_status = "temporal_incomplete"
                return 0

            print(
                f"Hybrid HQ stage 2/{stage_count}: MEMatte uncertain-edge refinement"
            )
            with profiler.stage("edge", frame_equivalents):
                edge_result = self._run_edge_stage(
                    points_list, parent_window, combined, temporal_model
                )
            self.propagated = bool(edge_result and temporal_propagated)
            if not edge_result:
                run_status = "edge_incomplete"
                return 0

            if evaluation_enabled:
                print("Hybrid HQ stage 3/3: no-reference evaluation and archive")
                with profiler.stage("evaluation", frame_equivalents):
                    self._run_evaluation_stage(
                        points_list, parent_window, combined, temporal_model
                    )
            run_status = "completed"
            print("Hybrid HQ matting completed")
            self._notify("matting_complete")
            return 1
        finally:
            if performance_enabled and profiler.stages:
                try:
                    report_path = write_performance_report(
                        os.path.join(core.temp_dir, "hybrid_performance"),
                        label=f"{temporal_model}_{memory_profile}",
                        metadata={
                            "status": run_status,
                            "model": "Hybrid HQ",
                            "temporal_model": temporal_model,
                            "memory_profile": memory_profile,
                            "start_frame": int(start_frame),
                            "end_frame": int(end_frame),
                            "objects": len(output_ids),
                            "matany_res": settings_mgr.get_session_setting(
                                "matany_res", 720
                            ),
                            "matany_chunk": settings_mgr.get_session_setting(
                                "matany_chunk", 16
                            ),
                            "mematte_tile_size": settings_mgr.get_session_setting(
                                "mematte_tile_size", 2048
                            ),
                            "mematte_max_tokens": settings_mgr.get_session_setting(
                                "mematte_max_tokens", 12000
                            ),
                            "hybrid_flow_resolution": settings_mgr.get_session_setting(
                                "hybrid_flow_resolution", 720
                            ),
                        },
                        profiler=profiler,
                        frame_equivalents=frame_equivalents,
                    )
                    print(f"Hybrid HQ performance: {report_path}")
                except Exception as exc:
                    print(f"Hybrid HQ performance report failed: {exc}")


# ---------------------------------------------------------------------------
# VideoMaMa backend
# ---------------------------------------------------------------------------

class VideoMaMaManager(MattingManager):
    """
    Matting manager that uses the VideoMaMa model.

    """

    BACKEND = "videomama"

    def __init__(self):
        super().__init__()
        self.pipeline = None

    def unload_matting_model(self):
        """Unload the VideoMaMa pipeline and free VRAM"""
        if self.pipeline is not None:
            try:
                if hasattr(self.pipeline, 'unet'):
                    self.pipeline.unet = None
                if hasattr(self.pipeline, 'vae'):
                    self.pipeline.vae = None
            except Exception as e:
                print(f"Warning during pipeline teardown: {e}")
            self.pipeline = None
        gc.collect()
        core.DeviceManager.clear_cache()
        print("Unloaded VideoMaMa model")
        
    def load_matting_model(self, load_to_cpu=False, parent_window=None):
        """
        Load the VideoMaMa pipeline.
        """
        from videomama.pipeline_svd_mask_numpy import VideoInferencePipeline

        device = self._prepare_device(load_to_cpu)
        settings_mgr = get_settings_manager()
        matting_model = settings_mgr.get_session_setting("matany_model", "VideoMaMa")
        max_size = settings_mgr.get_session_setting("matany_res", 0)
        overlap = settings_mgr.get_session_setting("matany_overlap", 2)
        batch_size = settings_mgr.get_session_setting("matany_chunk", 16)
        combined = settings_mgr.get_session_setting("matany_combined", False)

        if not ensure_models(["videomama", "svd_vae"], parent=parent_window):
            return False  # user cancelled or download failed

        try:
            self.pipeline = VideoInferencePipeline(
                base_model_path=os.path.join("checkpoints", "videomama"),
                unet_checkpoint_path=os.path.join("checkpoints", "videomama"),
                weight_dtype=torch.float16,
                device=str(device),
                enable_model_cpu_offload=False,    # Not much benefit here, since the vae is a small model
                vae_encode_chunk_size=1,          # Process VAE in small chunks, increasing doesnt help anything
                attention_mode="auto",            # Use xformers if available, else SDPA
                enable_vae_tiling=False,        # Tiling VAE is not worth it
                enable_vae_slicing=True,          # Process VAE one image at a time
            )
            print(f"Loaded {matting_model} model to {device} with max size {max_size}, overlap={overlap}, batch={batch_size}, and combined={combined}")
            return True

        except Exception as e:
            print(f"Error loading VideoMaMa pipeline: {e}")
            self.pipeline = None
            return False

    @torch.inference_mode()
    def run_matting(self, points_list, parent_window, combined=False):
        """
        Run matting on all frames, using tracked segmentation data.

        Args:
            points_list (list): List of point dictionaries containing object_id and frame information
            parent_window: Parent window for progress dialog
            combined (bool): If True, union all object masks into a single combined mask
                and run matting in a single pass instead of once per object.

        Returns:
            int: 1 if successful, 0 if cancelled/failed
        """

        if self.pipeline is None:
            print("VideoMaMa model not loaded")
            return 0

        core.DeviceManager.clear_cache()
        frame_count = core.VideoInfo.total_frames
        extension = core.get_frame_extension()
        settings_mgr = get_settings_manager()
        
        # --- VideoMaMa batch settings ---
        batch_size = settings_mgr.get_session_setting("matany_chunk", 16)   # frames per chunk sent to the model
        overlap = settings_mgr.get_session_setting("matany_overlap", 2)     # frames re-processed at each boundary for continuity
        if overlap == 0: 
            enable_boundary_blend = False
        else: 
            enable_boundary_blend = True # blend overlap frames linearly at chunk boundaries

        start_frame, end_frame, frames_to_process = self._get_frame_range()
        print(f"Processing matting from frame {start_frame} to {end_frame} ({frames_to_process} frames)")

        # Get unique object IDs from points list
        object_ids = sorted(list(set(point['object_id'] for point in points_list if 'object_id' in point)))
        if not object_ids:
            print("No objects found for matting")
            return 0

        # When combined mode is requested, run a single pass using the union of all
        # object masks loaded in memory — no files are written to disk.
        if combined and len(object_ids) > 1:
            combine_ids = object_ids
            object_ids = [0]
        else:
            combine_ids = None

        # Create matting directory if it doesn't exist
        os.makedirs(core.matting_dir, exist_ok=True)

        # If combined mode is selected, delete any existing matting files except object 0.
        if combined and os.path.exists(core.matting_dir):
            for frame_dirname in os.listdir(core.matting_dir):
                frame_dir = os.path.join(core.matting_dir, frame_dirname)
                if os.path.isdir(frame_dir):
                    for f in os.listdir(frame_dir):
                        if f != "0.png":
                            os.remove(os.path.join(frame_dir, f))

        # Calculate total operations for progress tracking
        total_operations = len(self._generate_windows(frames_to_process, batch_size, overlap)) * len(object_ids)

        progress_dialog, pbar = self._make_progress_dialog(parent_window, total_operations, unit="batch")
        operations_completed = 0

        try:
            # Process each object separately
            for object_id in object_ids:
                if progress_dialog.wasCanceled():
                    break

                pbar.set_description(f"Object {object_id}")
                print(f"Processing object {object_id}...")

                batches_completed = self._process_object(
                    object_id, start_frame, end_frame, overlap, batch_size, extension,
                    progress_dialog, pbar, operations_completed,
                    total_operations, parent_window, combine_ids=combine_ids,
                    enable_boundary_blend=enable_boundary_blend
                )

                if batches_completed is None:  # cancelled or failed
                    break

                operations_completed += batches_completed

        except Exception as e:
            pbar.close()
            progress_dialog.close()
            raise

        # Final cleanup
        pbar.close()
        core.DeviceManager.clear_cache()

        if progress_dialog.wasCanceled():
            print("Matting cancelled")
            self.propagated = False
            progress_dialog.close()
            return 0
        else:
            progress_dialog.setValue(100)
            if frame_count == frames_to_process:
                self.propagated = True  # only set propagated to True if the entire video was processed
            else:
                self.propagated = False
            print("Matting completed")
            self._notify('matting_complete')
            return 1

    def _load_batch(self, abs_start, abs_end, object_id, extension,
                    combine_ids=None, crop_rect=None):
        """
        Load a single batch window of frames and masks from disk, resized using
        the same proportional downscaling as MatAnyone (matany_res setting).
 
        If combine_ids is provided, masks for all IDs in that list are unioned
        in memory per frame rather than loading a single object mask from disk.
 
        If crop_rect is provided, both frames and masks are cropped to that region
        before being passed through _resize_image, so the model only processes the
        relevant portion of the frame.

        Args:
            abs_start:   Absolute start frame (inclusive)
            abs_end:     Absolute end frame (exclusive)
            object_id:   Object ID for mask lookup (ignored when combine_ids is set)
            extension:   Frame file extension
            combine_ids: Optional list of object IDs to union into a single mask
            crop_rect:   Optional (x1, y1, x2, y2) from core.compute_mask_bounding_box.
                         When set, frames and masks are cropped before resizing.
 
        Returns:
            tuple: (cond_frames, mask_frames, valid)
                - cond_frames: list of np.ndarray RGB uint8 (H, W, 3)
                - mask_frames: list of np.ndarray grayscale uint8 (H, W), binarized
                - valid: False if any source frame is missing
        """
        cond_frames = []
        mask_frames = []
 
        for frame_num in range(abs_start, abs_end):
            frame_path = os.path.join(core.frames_dir, f"{frame_num:05d}.{extension}")
            if not os.path.exists(frame_path):
                print(f"Warning: Frame not found: {frame_path}")
                return [], [], False
 
            # Load and convert frame to RGB, then proportionally downscale via matany_res
            frame = cv2.imread(frame_path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if crop_rect is not None:
                frame = core.apply_crop(frame, crop_rect)
            frame = self._resize_image(frame)
            resized_h, resized_w = frame.shape[:2]
 
            # Load mask — union across all combine_ids in memory, or single object from disk
            if combine_ids:
                union_mask = None
                for oid in combine_ids:
                    mask_path = os.path.join(core.mask_dir, f"{frame_num:05d}", f"{oid}.png")
                    if not os.path.exists(mask_path):
                        continue
                    m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    if m is None:
                        continue
                    union_mask = m if union_mask is None else np.maximum(union_mask, m)
                if union_mask is not None:
                    union_mask = core.apply_mask_postprocessing(union_mask)
                    if crop_rect is not None:
                        union_mask = core.apply_crop(union_mask, crop_rect)
                    mask = cv2.resize(union_mask, (resized_w, resized_h), interpolation=cv2.INTER_NEAREST)
                else:
                    mask = np.zeros((resized_h, resized_w), dtype=np.uint8)
            else:
                mask_path = os.path.join(core.mask_dir, f"{frame_num:05d}", f"{object_id}.png")
                if os.path.exists(mask_path):
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    mask = core.apply_mask_postprocessing(mask)
                    if crop_rect is not None:
                        mask = core.apply_crop(mask, crop_rect)
                    mask = cv2.resize(mask, (resized_w, resized_h), interpolation=cv2.INTER_NEAREST)
                else:
                    mask = np.zeros((resized_h, resized_w), dtype=np.uint8)
 
            # Binarize mask
            mask = (mask > 127).astype(np.uint8) * 255
 
            cond_frames.append(frame)
            mask_frames.append(mask)
 
        return cond_frames, mask_frames, True

    def _generate_windows(self, total_frames, batch_size, overlap):
        """
        Generate sliding window (start, end) pairs covering total_frames.
        end is exclusive. Overlap must be less than batch_size.

        Args:
            total_frames: Total frames to cover
            batch_size: Frames per batch 
            overlap: Overlap frames between consecutive batches

        Returns:
            list of (start, end) tuples (relative indices, end exclusive)
        """
        step = batch_size - overlap
        if step <= 0:
            print(f"Warning: overlap ({overlap}) >= batch_size ({batch_size}), clamping to batch_size - 1")
            overlap = batch_size - 1
            step = 1

        if total_frames <= batch_size:
            return [(0, total_frames)]

        windows = []
        pos = 0
        while pos < total_frames:
            end = min(pos + batch_size, total_frames)
            windows.append((pos, end))
            if end >= total_frames:
                break
            pos += step

        return windows


    def _process_object(self, object_id, start_frame, end_frame, overlap, batch_size, extension,
                        progress_dialog, pbar, operations_completed, total_operations, parent_window,
                        combine_ids=None, enable_boundary_blend=True):
        """
        Process a single object across all frames using windowed batching.
 
        Args:
            object_id: Object ID to process (also the output file label)
            start_frame: First frame (inclusive)
            end_frame: Last frame (inclusive)
            overlap: Overlap frames between batches
            batch_size: Number of frames per chunk sent to the model
            extension: Frame file extension
            progress_dialog: Qt progress dialog
            pbar: tqdm progress bar
            operations_completed: Batches completed before this object (for progress math)
            total_operations: Total batch count across all objects
            parent_window: Parent window for display updates
            combine_ids: If set, union masks for these IDs in memory rather than
                         loading a single object mask from disk
            enable_boundary_blend: Linearly blend overlap frames at chunk boundaries
 
        Returns:
            int: Number of batches completed, or None if cancelled/failed
        """
        frames_to_process = end_frame - start_frame + 1
        settings_mgr = get_settings_manager()
        display_update_frequency = settings_mgr.get_app_setting("display_update_frequency", 5)
 
        # Original frame dimensions for restoring output
        first_frame_path = os.path.join(core.frames_dir, f"{start_frame:05d}.{extension}")
        first_frame_img = cv2.imread(first_frame_path)
        if first_frame_img is not None:
            original_h, original_w = first_frame_img.shape[:2]
        else:
            original_w = core.VideoInfo.width
            original_h = core.VideoInfo.height

        # Compute a single crop rect covering all mask extents across the full frame range.
        # This is done once before any batch processing so every batch uses an identical
        # spatial region, keeping output consistent across chunk boundaries.
        object_ids_for_bbox = combine_ids if combine_ids is not None else [object_id]
        progress_dialog.setLabelText(f"Object {object_id}: Finding region of interest...")
        QApplication.processEvents()

        crop_rect = core.compute_mask_bounding_box(
            frame_range=range(start_frame, end_frame + 1),
            object_ids=object_ids_for_bbox,
            combine_ids=combine_ids,
        )

        if crop_rect is not None:
            cx1, cy1, cx2, cy2 = crop_rect
            print(f"Object {object_id}: crop rect ({cx1}, {cy1}) -> ({cx2}, {cy2})  "
                  f"[{cx2 - cx1 + 1}x{cy2 - cy1 + 1} of {original_w}x{original_h}]")
        else:
            print(f"Object {object_id}: Processing full frame")
 
        windows = self._generate_windows(frames_to_process, batch_size, overlap)
        batches_completed = 0

        # Stores soft alpha mattes (model working resolution, grayscale uint8) for the
        # last `overlap` frames of each batch.  Fed back as mask_frames for the overlap
        # frames of the next batch, giving the model continuity across chunk boundaries.
        # Kept soft (not binarized) so the model sees graduated edge information.
        previous_overlap_masks = None

        # Stores the last `overlap` committed alpha mattes at original resolution,
        # used for linear blending across the boundary after each non-first batch.
        prev_boundary_alphas = None
 
        for batch_idx, (window_start, window_end) in enumerate(windows):
            if progress_dialog.wasCanceled():
                return None
 
            abs_start = start_frame + window_start
            abs_end = start_frame + window_end  # exclusive
 
            # For non-first batches, the leading overlap frames are warm-up context only —
            # their output is discarded in favour of the already-committed result from the
            # previous batch. Only the non-overlap tail is saved.
            is_first_batch = (batch_idx == 0)
            output_start_offset = 0 if is_first_batch else overlap
 
            cond_frames, mask_frames, valid = self._load_batch(
                abs_start, abs_end, object_id, extension,
                combine_ids=combine_ids, crop_rect=crop_rect
            )
 
            if not valid:
                print(f"Warning: Skipping batch {batch_idx} for object {object_id} "
                    f"(frames {abs_start}-{abs_end - 1}) — missing data")
                pbar.update(1)
                batches_completed += 1
                # Reset both carry-overs so stale data is never injected after a gap
                previous_overlap_masks = None
                prev_boundary_alphas = None
                continue
 
            # --- FEEDBACK LOOP INJECTION ---
            # Replace the SAM2 masks for the overlap frames with the soft alpha mattes
            # predicted by the previous batch.  The model sees its own prior output as
            # guidance, encouraging consistent alpha values across the chunk boundary.
            if not is_first_batch and previous_overlap_masks is not None:
                for i in range(min(overlap, len(mask_frames), len(previous_overlap_masks))):
                    mask_frames[i] = previous_overlap_masks[i]
            # -------------------------------

            def _on_pipeline_progress(step, total, desc):
                progress_dialog.setLabelText(f"Batch {batch_idx + 1}/{len(windows)} — {desc}")
                QApplication.processEvents()
                if progress_dialog.wasCanceled():
                    raise RuntimeError("USER_CANCELLED")
 
            try:
                with torch.amp.autocast('cuda', enabled=False):
                    output_frames = self.pipeline.run(
                        cond_frames=cond_frames,
                        mask_frames=mask_frames,
                        seed=42,
                        progress_callback=_on_pipeline_progress,
                    )
            except RuntimeError as e:
                if str(e) == "USER_CANCELLED":
                    return None  # signals cancel upstream
                raise
            except Exception as e:
                print(f"Error in VideoMaMa inference for batch {batch_idx}: {e}")
                raise
 
            # --- CAPTURE SOFT MASKS FOR NEXT BATCH ---
            # Store the last `overlap` frames of the current output at model resolution
            # (before _restore_image_size) as soft grayscale — NOT binarized, so the
            # model receives graduated edge values rather than a hard binary boundary.
            previous_overlap_masks = []
            for frame_out in output_frames[-overlap:]:
                alpha_out = cv2.cvtColor(frame_out, cv2.COLOR_RGB2GRAY)
                previous_overlap_masks.append(alpha_out)
            # -----------------------------------------
 
            # Discard leading overlap output on non-first batches
            committed_output = output_frames[output_start_offset:]

            # --- LINEAR BLEND AT CHUNK BOUNDARIES ---
            # The overlap frames were re-processed by the new batch, giving us a second
            # prediction for those already-committed frames (output_frames[0:overlap]).
            # We linearly blend the original committed alpha (prev_boundary_alphas) with
            # the new prediction across the overlap window and re-write those frames.
            if enable_boundary_blend and not is_first_batch and prev_boundary_alphas is not None:
                for i in range(min(overlap, len(prev_boundary_alphas), len(output_frames))):
                    new_alpha = cv2.cvtColor(output_frames[i], cv2.COLOR_RGB2GRAY)
                    new_alpha = self._restore_image_size(new_alpha, (cx2 - cx1 + 1, cy2 - cy1 + 1) if crop_rect else (original_w, original_h))
                    if crop_rect is not None:
                        new_alpha = core.expand_to_full(new_alpha, crop_rect, original_w, original_h)
                    new_weight = (i + 1) / (overlap + 1)
                    blended = (
                        (1.0 - new_weight) * prev_boundary_alphas[i].astype(np.float32)
                        + new_weight * new_alpha.astype(np.float32)
                    ).clip(0, 255).astype(np.uint8)
                    abs_blend_frame = abs_start + i
                    mat_filename = os.path.join(core.matting_dir, f"{abs_blend_frame:05d}", f"{object_id}.png")
                    os.makedirs(os.path.dirname(mat_filename), exist_ok=True)
                    cv2.imwrite(mat_filename, blended)
            # -----------------------------------------

            # Collect the last `overlap` committed alphas (original resolution) for the
            # next batch's boundary blend, then write all committed frames to disk.
            current_boundary_alphas = []
            for i, frame_out in enumerate(committed_output):
                abs_frame = abs_start + output_start_offset + i
                alpha = cv2.cvtColor(frame_out, cv2.COLOR_RGB2GRAY)

                # Restore to the cropped region's pixel dimensions first, then expand
                # back into the full frame canvas so the saved matte is always full-size.
                if crop_rect is not None:
                    crop_w = cx2 - cx1 + 1
                    crop_h = cy2 - cy1 + 1
                    alpha = self._restore_image_size(alpha, (crop_w, crop_h))
                    final_alpha = core.expand_to_full(alpha, crop_rect, original_w, original_h)
                else:
                    final_alpha = self._restore_image_size(alpha, (original_w, original_h))

                if i >= len(committed_output) - overlap:
                    current_boundary_alphas.append(final_alpha.copy())
 
                mat_filename = os.path.join(core.matting_dir, f"{abs_frame:05d}", f"{object_id}.png")
                os.makedirs(os.path.dirname(mat_filename), exist_ok=True)
                cv2.imwrite(mat_filename, final_alpha)
 
                if abs_frame % display_update_frequency == 0:
                    try:
                        parent_window.frame_slider.setValue(abs_frame)
                        QApplication.processEvents()
                    except Exception:
                        pass

            prev_boundary_alphas = current_boundary_alphas if len(current_boundary_alphas) == overlap else None
 
            # Update progress
            pbar.update(1)
            batches_completed += 1
            current_progress = int(((operations_completed + batches_completed) * 100) / max(total_operations, 1))
            progress_dialog.setValue(min(current_progress, 99))
            QApplication.processEvents()
 
            core.DeviceManager.clear_cache()
 
        return batches_completed

# ---------------------------------------------------------------------------
# Manager Selection
# ---------------------------------------------------------------------------

def create_matting_manager() -> MattingManager:
    """
    Instantiate the correct matting manager based on the 'matting_model' session setting.
    """
    settings_mgr = get_settings_manager()
    matting_model = settings_mgr.get_session_setting("matany_model", "MatAnyone")
    if matting_model == "ViTMatte":
        return VitMatteManager()
    if matting_model == "MEMatte":
        return MematteManager()
    if matting_model == "Hybrid HQ":
        return HybridHQManager()
    if matting_model == "VideoMaMa":
        return VideoMaMaManager()
    return MatAnyManager()
