"""Project-owned exception types."""


class DatasetError(Exception):
    """Base exception for dataset pipeline failures."""


class ManifestError(DatasetError):
    """Raised when a source manifest has an invalid shape."""


class ImageProcessingError(DatasetError):
    """Raised when an image cannot be read, filtered, or exported."""


class ExternalToolError(DatasetError):
    """Raised when an external CLI dependency fails."""
