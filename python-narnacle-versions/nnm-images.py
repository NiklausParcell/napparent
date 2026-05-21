"""
NNM Images API - Neural Network Mask Image Processing API
Author: Niklaus Parcell
Date: 2024

Core functionality: Input images, output thermal overlay masks and pixel influence analysis
Based on the PsychicBarnacle algorithm from nnm_images.py
"""

import base64
import io
import json
import math
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.preprocessing import LabelEncoder


class NNMImagesAPI:
    """
    NewNikMath Images API for medical image analysis and thermal overlay generation.

    Core functionality:
    1. Train knowledge graphs on image data with labels
    2. Generate thermal overlay masks highlighting potentially malignant regions
    3. Calculate pixel influence values for image analysis
    """

    def __init__(self, image_size: int = 20, num_ranges: int = 5):
        """
        Initialize the NNM Images API processor.

        Args:
            image_size: Size of processed images (image_size x image_size)
            num_ranges: Number of pixel value ranges for knowledge graph
        """
        self.IMAGE_SIZE = image_size
        self.NUM_RANGES = num_ranges
        self.TOTAL_PIXELS = image_size * image_size

        # Knowledge graph: [i][j][pos][range] -> (sum, count)
        self.kg_sums = np.zeros(
            (image_size, image_size, 8, num_ranges), dtype=np.float32
        )
        self.kg_counts = np.zeros(
            (image_size, image_size, 8, num_ranges), dtype=np.int32
        )

        # Training data
        self.train_data = {"image_names": [], "targets": [], "image_to_target": {}}

        # Label encoding
        self.label_encoder = None
        self.label_mapping = {}

        # Threading
        self.num_threads = min(4, 8)  # Limit threads for API usage
        self.kg_lock = threading.Lock()

        # Model state
        self.is_trained = False

        # Streaming capabilities
        self.enable_streaming = False
        self.streaming_buffer = []
        self.streaming_buffer_size = 1000
        self.last_streaming_update = None
        self.training_stats = {
            "total_images": 0,
            "label_distribution": {},
            "training_time": 0.0,
        }

    def train(
        self,
        image_data: Union[List[np.ndarray], List[str]],
        target_data: Union[List[int], List[str]],
        image_labels: List[str] = None,
        temporal_data: Union[List, np.ndarray] = None,
        temporal_decay: float = 0.1,
    ) -> None:
        """
        Train the NNM Images model on image and target data.

        Args:
            image_data: List of image arrays or image file paths
            target_data: List of target values (any labels - will be encoded automatically)
            image_labels: List of image labels/names (optional)
            temporal_data: List of timestamps (optional)
            temporal_decay: Decay factor for temporal weighting
        """
        print(f"Training NNM Images model with {len(image_data)} images...")
        start_time = time.time()

        # Handle temporal data
        if temporal_data is not None:
            temporal_weights = self._calculate_temporal_weights(
                temporal_data, temporal_decay
            )
        else:
            temporal_weights = np.ones(len(image_data))

        # Encode labels automatically
        self.label_encoder = LabelEncoder()
        encoded_targets = self.label_encoder.fit_transform(target_data)

        # Create label mapping for reference
        unique_labels = self.label_encoder.classes_
        self.label_mapping = {i: label for i, label in enumerate(unique_labels)}

        # Calculate label distribution
        label_counts = np.bincount(encoded_targets)
        self.training_stats["label_distribution"] = {
            self.label_mapping[i]: int(count) for i, count in enumerate(label_counts)
        }

        print(f"Label mapping: {self.label_mapping}")
        print(f"Label distribution: {self.training_stats['label_distribution']}")

        # Process images and build knowledge graph

        def process_image_batch(batch_indices):
            for idx in batch_indices:
                try:
                    # Load image
                    if isinstance(image_data[idx], str):
                        # File path
                        image = self._load_image_from_path(image_data[idx])
                    else:
                        # Image array
                        image = self._process_image_array(image_data[idx])

                    # Get encoded label (already processed above)
                    encoded_label = encoded_targets[idx]

                    # Process surrounding pixels with temporal weighting
                    self._process_surrounding_pixels(
                        image, encoded_label, temporal_weights[idx]
                    )

                except Exception as e:
                    print(f"Error processing image {idx}: {e}")
                    continue

        # Process images in batches
        batch_size = min(self.num_threads, 4)
        indices = list(range(len(image_data)))

        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            for i in range(0, len(indices), batch_size):
                batch_indices = indices[i : i + batch_size]
                executor.submit(process_image_batch, batch_indices)

        # Update training stats
        self.training_stats.update(
            {"total_images": len(image_data), "training_time": time.time() - start_time}
        )

        self.is_trained = True

        print(f"✓ Training complete in {self.training_stats['training_time']:.1f}s")
        print(f"Label distribution: {self.training_stats['label_distribution']}")

    def _calculate_temporal_weights(
        self, temporal_data: Union[List, np.ndarray], temporal_decay: float
    ) -> np.ndarray:
        """Calculate temporal weights for training data."""
        temporal_series = np.array(temporal_data)

        # Convert to datetime if not already
        if not np.issubdtype(temporal_series.dtype, np.datetime64):
            temporal_series = np.array([np.datetime64(t) for t in temporal_series])

        # Calculate temporal weights (more recent = higher weight)
        max_time = np.max(temporal_series)
        time_diffs = (max_time - temporal_series) / np.timedelta64(
            1, "D"
        )  # Convert to days
        temporal_weights = np.exp(-temporal_decay * time_diffs)

        return temporal_weights

    def _load_image_from_path(self, image_path: str) -> np.ndarray:
        """Load and process image from file path."""
        try:
            with Image.open(image_path) as img:
                # Convert to grayscale and resize
                img_gray = img.convert("L")
                img_resized = img_gray.resize(
                    (self.IMAGE_SIZE, self.IMAGE_SIZE), Image.LANCZOS
                )

                # Convert to numpy array and normalize
                image_array = np.array(img_resized, dtype=np.float32) / 255.0
                return image_array
        except Exception as e:
            print(f"Warning: Failed to load {image_path}: {e}")
            # Fallback to synthetic image
            seed = hash(image_path) % 10000
            return self._generate_synthetic_image(seed)

    def _process_image_array(self, image_array: np.ndarray) -> np.ndarray:
        """Process image array to standard format."""
        # Ensure 2D grayscale
        if len(image_array.shape) == 3:
            # Convert RGB to grayscale
            image_array = np.mean(image_array, axis=2)

        # Resize if needed
        if image_array.shape != (self.IMAGE_SIZE, self.IMAGE_SIZE):
            from PIL import Image

            img = Image.fromarray((image_array * 255).astype(np.uint8))
            img_resized = img.resize((self.IMAGE_SIZE, self.IMAGE_SIZE), Image.LANCZOS)
            image_array = np.array(img_resized, dtype=np.float32) / 255.0

        # Normalize to 0-1 range
        image_array = np.clip(image_array, 0.0, 1.0)

        return image_array

    def _generate_synthetic_image(self, seed: int = 42) -> np.ndarray:
        """Generate a synthetic test image."""
        np.random.seed(seed)
        image = np.zeros((self.IMAGE_SIZE, self.IMAGE_SIZE), dtype=np.float32)

        center = self.IMAGE_SIZE // 2
        for i in range(self.IMAGE_SIZE):
            for j in range(self.IMAGE_SIZE):
                distance = math.sqrt((i - center) ** 2 + (j - center) ** 2)
                base_value = 0.5 + 0.3 * math.sin(distance * 0.3)
                noise = np.random.random() * 0.2
                image[i, j] = max(0.0, min(1.0, base_value + noise))

        return image

    def _process_surrounding_pixels(
        self, image: np.ndarray, image_label: int, temporal_weight: float = 1.0
    ):
        """
        Process surrounding pixels and update knowledge graph.

        Args:
            image: Input image
            image_label: Label for the image (0=benign, 1=malignant)
            temporal_weight: Temporal weighting factor
        """
        # 8 surrounding positions
        dx = [-1, -1, -1, 0, 0, 1, 1, 1]
        dy = [-1, 0, 1, -1, 1, -1, 0, 1]

        for i in range(self.IMAGE_SIZE):
            for j in range(self.IMAGE_SIZE):
                for pos in range(8):
                    x = i + dx[pos]
                    y = j + dy[pos]

                    if 0 <= x < self.IMAGE_SIZE and 0 <= y < self.IMAGE_SIZE:
                        pixel_value = np.clip(image[x, y], 0.0, 1.0)
                        range_index = min(
                            int(pixel_value * self.NUM_RANGES), self.NUM_RANGES - 1
                        )

                        # Thread-safe update with temporal weighting
                        with self.kg_lock:
                            self.kg_sums[i, j, pos, range_index] += (
                                float(image_label) * temporal_weight
                            )
                            self.kg_counts[i, j, pos, range_index] += 1

    def predict_image_mask(self, image: Union[np.ndarray, str]) -> np.ndarray:
        """
        Generate thermal mask for an image.

        Args:
            image: Input image array or file path

        Returns:
            Generated thermal mask as numpy array
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        # Load/process image
        if isinstance(image, str):
            processed_image = self._load_image_from_path(image)
        else:
            processed_image = self._process_image_array(image)

        # Generate mask from knowledge graph
        mask = self._generate_mask_from_kg(processed_image)

        return mask

    def _generate_mask_from_kg(self, image: np.ndarray) -> np.ndarray:
        """
        Generate mask from knowledge graph.

        Args:
            image: Input image

        Returns:
            Generated mask
        """
        mask = np.zeros((self.IMAGE_SIZE, self.IMAGE_SIZE), dtype=np.float32)

        # 8 surrounding positions
        dx = [-1, -1, -1, 0, 0, 1, 1, 1]
        dy = [-1, 0, 1, -1, 1, -1, 0, 1]

        for i in range(self.IMAGE_SIZE):
            for j in range(self.IMAGE_SIZE):
                total_weighted_sum = 0.0
                total_count = 0

                for pos in range(8):
                    x = i + dx[pos]
                    y = j + dy[pos]

                    if 0 <= x < self.IMAGE_SIZE and 0 <= y < self.IMAGE_SIZE:
                        pixel_value = np.clip(image[x, y], 0.0, 1.0)
                        range_index = min(
                            int(pixel_value * self.NUM_RANGES), self.NUM_RANGES - 1
                        )

                        range_sum = self.kg_sums[i, j, pos, range_index]
                        range_count = self.kg_counts[i, j, pos, range_index]

                        if range_count > 0:
                            range_avg = range_sum / range_count
                            total_weighted_sum += range_avg
                            total_count += 1

                if total_count > 0:
                    mask[i, j] = abs(total_weighted_sum / total_count)

        # Normalize mask values
        max_val = np.max(mask)
        if max_val > 0:
            if max_val < 0.1:
                mask = mask / max_val
            elif max_val < 0.5:
                boost_factor = 0.5 / max_val
                mask = np.minimum(1.0, mask * boost_factor)

        return mask

    def predict_pixel_influences(
        self, image: Union[np.ndarray, str]
    ) -> Dict[str, Union[np.ndarray, Dict]]:
        """
        Calculate pixel influence values for an image.

        Args:
            image: Input image array or file path

        Returns:
            Dictionary with mask, statistics, and influence data
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        # Load/process image
        if isinstance(image, str):
            processed_image = self._load_image_from_path(image)
        else:
            processed_image = self._process_image_array(image)

        # Generate mask
        mask = self._generate_mask_from_kg(processed_image)

        # Calculate statistics
        max_val = np.max(mask)
        min_val = np.min(mask)
        non_zero_pixels = np.sum(mask > 0)
        avg_val = np.mean(mask[mask > 0]) if non_zero_pixels > 0 else 0

        # Find high-influence regions
        high_influence_threshold = avg_val + (max_val - avg_val) * 0.5
        high_influence_pixels = np.sum(mask > high_influence_threshold)

        return {
            "mask": mask,
            "statistics": {
                "min_value": float(min_val),
                "max_value": float(max_val),
                "average_value": float(avg_val),
                "non_zero_pixels": int(non_zero_pixels),
                "total_pixels": int(self.TOTAL_PIXELS),
                "high_influence_pixels": int(high_influence_pixels),
                "high_influence_threshold": float(high_influence_threshold),
            },
            "original_image": processed_image,
        }

    def create_thermal_overlay(
        self, image: Union[np.ndarray, str]
    ) -> Dict[str, Union[np.ndarray, str]]:
        """
        Create thermal overlay visualization for an image.

        Args:
            image: Input image array or file path

        Returns:
            Dictionary with thermal overlay and base64 encoded image
        """
        # Get pixel influences
        result = self.predict_pixel_influences(image)
        original_image = result["original_image"]
        mask = result["mask"]

        # Create thermal overlay
        thermal_overlay = self._create_thermal_overlay_image(original_image, mask)

        # Convert to base64 for API response
        thermal_b64 = self._image_to_base64(thermal_overlay)

        return {
            "thermal_overlay": thermal_overlay,
            "thermal_overlay_b64": thermal_b64,
            "mask": mask,
            "statistics": result["statistics"],
        }

    def _create_thermal_overlay_image(
        self, original_image: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        """
        Create thermal overlay image.

        Args:
            original_image: Original image
            mask: Generated mask

        Returns:
            Thermal overlay image as RGB array
        """
        height, width = original_image.shape
        overlay = np.zeros((height, width, 3), dtype=np.uint8)

        for i in range(height):
            for j in range(width):
                original_intensity = original_image[i, j]
                mask_intensity = mask[i, j]
                overlay[i, j] = self._get_thermal_color(
                    original_intensity, mask_intensity
                )

        return overlay

    def _get_thermal_color(
        self, original_intensity: float, mask_intensity: float
    ) -> Tuple[int, int, int]:
        """
        Get thermal color for overlay.

        Args:
            original_intensity: Original image intensity (0-1)
            mask_intensity: Mask intensity (0-1)

        Returns:
            RGB color tuple
        """
        base_gray = int(original_intensity * 255)
        mask_intensity = max(0.0, min(1.0, mask_intensity))

        if mask_intensity < 0.25:
            # Very low: pure grayscale
            return (base_gray, base_gray, base_gray)
        elif mask_intensity < 0.5:
            # Low to medium: Green
            factor = (mask_intensity - 0.25) / 0.25
            red = base_gray
            green = int(base_gray + factor * (255 - base_gray))
            blue = base_gray
            return (red, green, blue)
        elif mask_intensity < 0.75:
            # Medium to high: Yellow
            factor = (mask_intensity - 0.5) / 0.25
            red = int(base_gray + factor * (255 - base_gray))
            green = 255
            blue = base_gray
            return (red, green, blue)
        else:
            # High to very high: Orange to Red
            factor = (mask_intensity - 0.75) / 0.25
            red = 255
            green = int(255 * (1.0 - factor * 0.6))  # Orange to red transition
            blue = base_gray
            return (red, green, blue)

    def _image_to_base64(self, image_array: np.ndarray) -> str:
        """Convert image array to base64 string."""
        try:
            # Convert to PIL Image
            if len(image_array.shape) == 3:
                # RGB image
                pil_image = Image.fromarray(image_array, "RGB")
            else:
                # Grayscale image
                pil_image = Image.fromarray((image_array * 255).astype(np.uint8), "L")

            # Convert to base64
            buffer = io.BytesIO()
            pil_image.save(buffer, format="PNG")
            img_str = base64.b64encode(buffer.getvalue()).decode()

            return img_str
        except Exception as e:
            print(f"Error converting image to base64: {e}")
            return ""

    def save_knowledge_graph(self, filename: str = None) -> str:
        """
        Save knowledge graph to JSON file.

        Args:
            filename: Optional filename (auto-generated if None)

        Returns:
            Filename of saved knowledge graph
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving knowledge graph")

        # Generate filename if not provided
        if filename is None:
            now = datetime.now()
            filename = f"KG_{now.month:02d}{now.day:02d}{now.year%100:02d}{now.hour:02d}{now.minute:02d}.json"

        # Prepare data for JSON serialization
        kg_data = {
            "metadata": {
                "image_size": self.IMAGE_SIZE,
                "num_ranges": self.NUM_RANGES,
                "total_pixels": self.TOTAL_PIXELS,
                "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "training_stats": self.training_stats,
            },
            "label_encoder": {
                "classes": self.label_encoder.classes_.tolist()
                if self.label_encoder
                else [],
                "label_mapping": self.label_mapping,
            },
            "knowledge_graph": {},
        }

        # Convert numpy arrays to lists for JSON serialization
        for i in range(self.IMAGE_SIZE):
            for j in range(self.IMAGE_SIZE):
                pixel_key = f"{i}_{j}"
                kg_data["knowledge_graph"][pixel_key] = {}

                for pos in range(8):
                    pos_key = f"pos_{pos}"
                    kg_data["knowledge_graph"][pixel_key][pos_key] = {}

                    for range_idx in range(self.NUM_RANGES):
                        range_key = f"range_{range_idx}"
                        range_sum = float(self.kg_sums[i, j, pos, range_idx])
                        range_count = int(self.kg_counts[i, j, pos, range_idx])
                        range_avg = range_sum / range_count if range_count > 0 else 0.0

                        kg_data["knowledge_graph"][pixel_key][pos_key][range_key] = {
                            "sum": range_sum,
                            "count": range_count,
                            "average": range_avg,
                        }

        # Save to JSON file
        try:
            with open(filename, "w") as file:
                json.dump(kg_data, file, indent=2)
            print(f"✓ Knowledge graph saved to: {filename}")
            return filename
        except Exception as e:
            print(f"Error saving knowledge graph: {e}")
            return ""

    def load_knowledge_graph(self, filename: str) -> None:
        """
        Load knowledge graph from JSON file.

        Args:
            filename: Knowledge graph filename
        """
        if not filename.endswith(".json"):
            filename += ".json"

        if not Path(filename).exists():
            raise FileNotFoundError(f"Knowledge graph file not found: {filename}")

        print(f"Loading knowledge graph from: {filename}")

        try:
            with open(filename, "r") as file:
                kg_data = json.load(file)

            # Load metadata
            metadata = kg_data.get("metadata", {})
            self.IMAGE_SIZE = metadata.get("image_size", self.IMAGE_SIZE)
            self.NUM_RANGES = metadata.get("num_ranges", self.NUM_RANGES)
            self.TOTAL_PIXELS = metadata.get("total_pixels", self.TOTAL_PIXELS)
            self.training_stats = metadata.get("training_stats", self.training_stats)

            # Load label encoder
            label_encoder_data = kg_data.get("label_encoder", {})
            if label_encoder_data:
                self.label_encoder = LabelEncoder()
                self.label_encoder.classes_ = np.array(
                    label_encoder_data.get("classes", [])
                )
                self.label_mapping = label_encoder_data.get("label_mapping", {})

            # Reset knowledge graph
            self.kg_sums = np.zeros(
                (self.IMAGE_SIZE, self.IMAGE_SIZE, 8, self.NUM_RANGES), dtype=np.float32
            )
            self.kg_counts = np.zeros(
                (self.IMAGE_SIZE, self.IMAGE_SIZE, 8, self.NUM_RANGES), dtype=np.int32
            )

            # Load data
            kg = kg_data["knowledge_graph"]
            loaded_entries = 0

            for pixel_key, pixel_data in kg.items():
                i, j = map(int, pixel_key.split("_"))

                for pos_key, pos_data in pixel_data.items():
                    pos = int(pos_key.split("_")[1])

                    for range_key, range_data in pos_data.items():
                        range_idx = int(range_key.split("_")[1])

                        if (
                            0 <= i < self.IMAGE_SIZE
                            and 0 <= j < self.IMAGE_SIZE
                            and 0 <= pos < 8
                            and 0 <= range_idx < self.NUM_RANGES
                        ):
                            self.kg_sums[i, j, pos, range_idx] = range_data["sum"]
                            self.kg_counts[i, j, pos, range_idx] = range_data["count"]

                            if range_data["count"] > 0:
                                loaded_entries += 1

            self.is_trained = True
            print(f"✓ Loaded knowledge graph with {loaded_entries} non-zero entries")

        except Exception as e:
            print(f"Error loading knowledge graph: {e}")
            raise

    def get_model_info(self) -> Dict:
        """Get information about the trained model."""
        if not self.is_trained:
            return {"trained": False}

        total_entries = np.sum(self.kg_counts > 0)

        return {
            "trained": True,
            "image_size": self.IMAGE_SIZE,
            "num_ranges": self.NUM_RANGES,
            "total_pixels": self.TOTAL_PIXELS,
            "knowledge_graph_entries": int(total_entries),
            "label_mapping": self.label_mapping,
            "training_stats": self.training_stats,
        }

    def enable_streaming_mode(self, buffer_size: int = 1000) -> None:
        """
        Enable streaming mode for incremental learning.

        Args:
            buffer_size: Maximum number of samples to buffer before processing
        """
        self.enable_streaming = True
        self.streaming_buffer_size = buffer_size
        self.streaming_buffer = []
        print(f"Image streaming mode enabled with buffer size: {buffer_size}")

    def disable_streaming_mode(self) -> None:
        """Disable streaming mode and process any remaining buffered data."""
        if self.enable_streaming and self.streaming_buffer:
            print(
                f"Processing {len(self.streaming_buffer)} remaining image samples from buffer"
            )
            self._process_image_streaming_buffer()
        self.enable_streaming = False
        self.streaming_buffer = []
        print("Image streaming mode disabled")

    def add_streaming_image_data(
        self,
        image_data: Union[List[np.ndarray], List[str]],
        target_data: Union[List[int], List[str]],
        temporal_data: Union[List, pd.Series] = None,
        temporal_decay: float = 0.1,
    ) -> Dict[str, any]:
        """
        Add new image data to the model in streaming mode.

        Args:
            image_data: New image data to add
            target_data: Target data for new samples
            temporal_data: Temporal data for new samples
            temporal_decay: Decay factor for temporal weighting

        Returns:
            Dictionary with streaming update statistics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before adding streaming data")

        if not self.enable_streaming:
            raise ValueError(
                "Streaming mode must be enabled before adding streaming data"
            )

        # Process temporal data
        if temporal_data is not None:
            if isinstance(temporal_data, list):
                temporal_series = pd.to_datetime(temporal_data)
            else:
                temporal_series = pd.to_datetime(temporal_data)
        else:
            # Use current time if no temporal data provided
            temporal_series = pd.Series([pd.Timestamp.now()] * len(image_data))

        # Calculate temporal weights
        max_time = temporal_series.max()
        temporal_weights = np.exp(
            -temporal_decay * (max_time - temporal_series).dt.total_seconds() / 86400
        )

        # Add to streaming buffer
        streaming_sample = {
            "image_data": image_data,
            "target_data": target_data,
            "temporal_weights": temporal_weights,
            "timestamp": pd.Timestamp.now(),
        }

        self.streaming_buffer.append(streaming_sample)

        # Process buffer if it's full
        if len(self.streaming_buffer) >= self.streaming_buffer_size:
            return self._process_image_streaming_buffer()
        else:
            return {
                "status": "buffered",
                "buffer_size": len(self.streaming_buffer),
                "buffer_capacity": self.streaming_buffer_size,
                "samples_added": len(image_data),
            }

    def _process_image_streaming_buffer(self) -> Dict[str, any]:
        """Process the image streaming buffer and update the model."""
        if not self.streaming_buffer:
            return {"status": "no_data", "samples_processed": 0}

        print(
            f"Processing image streaming buffer with {len(self.streaming_buffer)} samples"
        )

        # Combine all buffered data
        all_images = []
        all_targets = []
        all_temporal_weights = []

        for sample in self.streaming_buffer:
            all_images.extend(sample["image_data"])
            all_targets.extend(sample["target_data"])
            all_temporal_weights.extend(sample["temporal_weights"])

        # Update knowledge graph with new data
        self._update_image_knowledge_graph(
            all_images, all_targets, all_temporal_weights
        )

        # Update model statistics
        self._update_image_model_statistics(all_targets, all_temporal_weights)

        # Clear buffer
        samples_processed = len(self.streaming_buffer)
        self.streaming_buffer = []
        self.last_streaming_update = pd.Timestamp.now()

        return {
            "status": "processed",
            "samples_processed": samples_processed,
            "buffer_size": 0,
            "timestamp": self.last_streaming_update.isoformat(),
        }

    def _update_image_knowledge_graph(
        self, image_data: List, target_data: List, temporal_weights: List
    ) -> None:
        """Update image knowledge graph with new streaming data."""
        # Process each image
        for image, label, weight in zip(image_data, target_data, temporal_weights):
            # Load and preprocess image
            if isinstance(image, str):
                image_array = self._load_image(image)
            else:
                image_array = image

            # Resize image
            image_array = self._resize_image(image_array)

            # Encode label
            image_label = self._encode_label(label)

            # Update knowledge graph with temporal weighting
            self._process_surrounding_pixels(image_array, image_label, weight)

    def _update_image_model_statistics(
        self, target_data: List, temporal_weights: List
    ) -> None:
        """Update image model statistics with new streaming data."""
        # Update training stats
        self.training_stats["total_images"] += len(target_data)

        # Update label distribution
        for label, weight in zip(target_data, temporal_weights):
            if label not in self.training_stats["label_distribution"]:
                self.training_stats["label_distribution"][label] = 0
            self.training_stats["label_distribution"][label] += weight

    def get_streaming_status(self) -> Dict[str, any]:
        """Get current image streaming status and statistics."""
        return {
            "streaming_enabled": self.enable_streaming,
            "buffer_size": len(self.streaming_buffer),
            "buffer_capacity": self.streaming_buffer_size,
            "last_update": self.last_streaming_update.isoformat()
            if self.last_streaming_update
            else None,
            "total_images": self.training_stats["total_images"],
            "label_distribution": self.training_stats["label_distribution"],
        }


# Convenience functions for direct API usage
def create_nnm_images_model(image_size: int = 20, num_ranges: int = 5) -> NNMImagesAPI:
    """Create a new NNM Images model instance."""
    return NNMImagesAPI(image_size, num_ranges)


def train_nnm_images_model(
    image_data: Union[List[np.ndarray], List[str]],
    target_data: Union[List[int], List[str]],
    image_size: int = 20,
    num_ranges: int = 5,
    image_labels: List[str] = None,
    temporal_data: Union[List, np.ndarray] = None,
    temporal_decay: float = 0.1,
) -> NNMImagesAPI:
    """
    Train a new NNM Images model and return it.

    Args:
        image_data: List of image arrays or image file paths
        target_data: List of target values (any labels - will be encoded automatically)
        image_size: Size of processed images
        num_ranges: Number of pixel value ranges
        image_labels: List of image labels/names (optional)
        temporal_data: List of timestamps (optional)
        temporal_decay: Decay factor for temporal weighting

    Returns:
        Trained NNMImagesAPI instance
    """
    model = NNMImagesAPI(image_size, num_ranges)
    model.train(image_data, target_data, image_labels, temporal_data, temporal_decay)
    return model


def analyze_image(
    model: NNMImagesAPI, image: Union[np.ndarray, str]
) -> Dict[str, Union[np.ndarray, Dict, str]]:
    """
    Analyze image using a trained NNM model.

    Args:
        model: Trained NNMImagesAPI instance
        image: Image array or file path to analyze

    Returns:
        Dictionary with thermal overlay, mask, and analysis data
    """
    thermal_result = model.create_thermal_overlay(image)
    pixel_influences = model.predict_pixel_influences(image)

    return {
        "thermal_overlay": thermal_result["thermal_overlay"],
        "thermal_overlay_b64": thermal_result["thermal_overlay_b64"],
        "mask": thermal_result["mask"],
        "statistics": thermal_result["statistics"],
        "pixel_influences": pixel_influences,
    }
