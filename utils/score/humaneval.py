import contextlib
import signal


def compute_score(solution_str, test_code_str, timeout=5) -> float:
    retval = 0.0
    string_in_code_block = code_block_string(solution_str)
    program = f"""{string_in_code_block}\n{test_code_str}"""
    try:
        with time_limit(timeout):
            exec(program)
        retval = 1.0
    except TimeoutError:
        retval = 0.0
    except BaseException as e:
        retval = 0.0
    return retval


def code_block_string(solution_str):
    return solution_str.split("```")[0]


@contextlib.contextmanager
def time_limit(seconds: float):
    def signal_handler(signum, frame):
        raise TimeoutError("Timed out!")

    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
