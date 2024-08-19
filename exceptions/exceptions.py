class BaseBCLRException(Exception):
    """Base exception"""


class InvalidBackboneError(BaseBCLRException):
    """Raised when the choice of backbone Convnet is invalid."""


class InvalidDatasetSelection(BaseBCLRException):
    """Raised when the choice of dataset is invalid."""

