"""Project-wide configuration for multi-modal food processing."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    """Simple, extensible configuration container."""

    DATA_RAW_DIR: Path = Path("data/raw")
    DATA_PROCESSED_DIR: Path = Path("data/processed")

    VIDEO_NUM_FRAMES: int = 20
    VIDEO_HEIGHT: int = 64
    VIDEO_WIDTH: int = 64
    VIDEO_CHANNELS: int = 1

    RHEOLOGY_TIME_STEPS: int = 100

    NUM_CLASSES: int = 3  # low / medium / high viscosity
    NUM_REG_TARGETS: int = 2  # e.g., viscosity parameters

    BATCH_SIZE: int = 32
    LEARNING_RATE: float = 1e-3
    NUM_EPOCHS: int = 30

    def raw_video_dir(self) -> Path:
        return self.DATA_RAW_DIR / "videos"

    def raw_rheology_dir(self) -> Path:
        return self.DATA_RAW_DIR / "rheology"

    def raw_microscopy_dir(self) -> Path:
        return self.DATA_RAW_DIR / "microscopy"

    def ensure_dirs(self) -> None:
        """Create standard raw/processed subdirectories."""
        for path in [
            self.DATA_RAW_DIR,
            self.raw_video_dir(),
            self.raw_rheology_dir(),
            self.raw_microscopy_dir(),
            self.DATA_PROCESSED_DIR,
        ]:
            Path(path).mkdir(parents=True, exist_ok=True)


# Default instance used throughout the project.
config = Config()
