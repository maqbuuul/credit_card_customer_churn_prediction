import sys
from src.logger import logging


def error_message_detail(error, error_detail: sys):
    """
    Generate detailed error message with file name and line number

    Args:
        error: The error object
        error_detail: sys exception details

    Returns:
        str: Formatted error message with filename and line number
    """
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno

    error_message = f"Error occurred in Python script: [{file_name}] at line [{line_number}] error message: [{str(error)}]"

    return error_message


class CustomException(Exception):
    """
    Custom exception class that provides detailed error information
    including file name and line number where error occurred
    """

    def __init__(self, error_message, error_detail: sys):
        """
        Initialize custom exception with error message and details

        Args:
            error_message: The error message
            error_detail: sys exception details
        """
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self):
        """String representation of the exception"""
        return self.error_message
