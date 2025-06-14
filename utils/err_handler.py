import os
import signal
import sys

import ipdb
import requests
import torch
import torch.distributed as dist
from dotenv import load_dotenv

load_dotenv()


class ErrorHandler:
    def __init__(self):
        self.debug_mode = (
            os.environ.get("DEBUG_MODE", "False").lower() == "true"
        )
        self.bot_token = os.environ.get("TG_BOT_TOKEN")
        self.chat_id = os.environ.get("TG_CHAT_ID")

        if not self.bot_token or not self.chat_id:
            print(
                "Warning: Telegram credentials not found in environment variables"
            )
            self.telegram_enabled = False
        else:
            self.telegram_enabled = True
            self.telegram_url = (
                f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            )

        self.local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))

        self.in_debug_session = False  # Add this line

    def init(self):
        if self.local_rank == 0:
            sys.excepthook = self.excepthook
            signal.signal(signal.SIGINT, self.signal_handler)
            print(
                f"Error handler set up for main process. Debug mode: {'ON' if self.debug_mode else 'OFF'}"
            )
        else:
            sys.excepthook = self.non_main_excepthook
            print("Error handler set up for non-main process.")

    def send_telegram_notification(self, message):
        if not self.telegram_enabled:
            print("Telegram notifications disabled: missing credentials")
            return

        payload = {"chat_id": self.chat_id, "text": message}
        try:
            requests.post(self.telegram_url, json=payload)
        except Exception as e:
            print(f"Failed to send Telegram notification: {e}")

    def excepthook(self, type, value, traceback):
        error_message = f"An error occurred: {value}"
        print(error_message)

        if self.local_rank == 0:
            self.send_telegram_notification(error_message)
            if self.debug_mode and not self.in_debug_session:
                print("Main process entering debugging environment...")
                self.in_debug_session = True  # Set debug flag
                self.notify_other_processes()
                ipdb.post_mortem(traceback)
                self.in_debug_session = False  # Reset debug flag
            else:
                self.terminate_all_processes()

    def non_main_excepthook(self, type, value, traceback):
        print(f"Error in process {self.local_rank}: {value}")
        # Wait for main process to enter debugging environment
        self.wait_for_main_process()

    def notify_other_processes(self):
        if dist.is_initialized():
            dist.broadcast(torch.tensor([1], device="cuda"), src=0)

    def wait_for_main_process(self):
        if dist.is_initialized():
            tensor = torch.tensor([0], device="cuda")
            dist.broadcast(tensor, src=0)
            if tensor.item() == 1:
                print(
                    f"Process {self.local_rank} received debug signal, exiting..."
                )
                sys.exit(0)

    def terminate_all_processes(self):
        if dist.is_initialized():
            dist.broadcast(torch.tensor([2], device="cuda"), src=0)
        sys.exit(1)

    def signal_handler(self, signum, frame):
        print("Interrupt received, exiting...")
        sys.exit(0)


# Create a global instance of ErrorHandler
error_handler = ErrorHandler()


def init_err_handler():
    error_handler.init()


# Example usage
if __name__ == "__main__":
    init_err_handler()
    # Your main program code here
    # For example, to test the error handler:
    raise Exception("Test error")
