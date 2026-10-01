"""Controlled error types shared by the backend and the FastAPI layer."""


class TripMateError(Exception):
    """Base class: carries a stable error code, an HTTP status and a safe message."""

    code = "internal_error"
    status_code = 500
    default_message = "Something went wrong while processing the request."

    def __init__(self, message: str | None = None):
        self.message = message or self.default_message
        super().__init__(self.message)


class InvalidRequestError(TripMateError):
    code = "invalid_request"
    status_code = 400
    default_message = "The request is not valid."


class ThreadNotFoundError(TripMateError):
    code = "thread_not_found"
    status_code = 404
    default_message = "No conversation was found for this thread_id."


class NoPendingApprovalError(TripMateError):
    code = "no_pending_approval"
    status_code = 409
    default_message = "There is no draft waiting for approval on this thread."


class DependencyUnavailableError(TripMateError):
    code = "dependency_unavailable"
    status_code = 503
    default_message = "A required service is temporarily unavailable. Please try again."


class MalformedToolResponse(Exception):
    """An MCP tool returned something that could not be parsed as expected."""
