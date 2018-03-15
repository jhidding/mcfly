from noodles import gather_all, schedule
from noodles.tutorial import (sub, mul, add)
from mcfly.run_noodles import run_remote
from echo_add import echo_add


def test_xenon_42_multi():
    A = echo_add(1, 1)
    B = sub(3, A)

    multiples = [mul(echo_add(i, B), A) for i in range(6)]
    C = schedule(sum)(gather_all(multiples))
    return C


if __name__ == "__main__":
    result = run_remote(test_xenon_42_multi())
    print("The answer is:", result)
