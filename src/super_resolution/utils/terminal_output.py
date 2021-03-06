### terminal_output.py ###
# This module contains the TColors class for making colored Error messages in the terminal.
# (from: https://stackoverflow.com/questions/287871/how-to-print-colored-text-to-the-terminal)


# colors for terminal output
class TColors:
    HEADER = '\033[95m'
    NOTE = '\033[90m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'