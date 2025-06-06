import time


class TimeLogger:
    """Logs execution time for key steps."""
    @staticmethod
    def log(message, start_time):
        time_spent = time.time() - start_time
        print(f"Time spent: [{time_spent:.2f}]sec - {message}")
