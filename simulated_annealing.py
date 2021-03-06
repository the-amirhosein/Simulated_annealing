from numpy import asarray, exp
from numpy.random import randn, rand, seed

def objective(x):
    return -3 * x[0] ** 5 + 3 * x[0] ** 4 + 5 * x[0] ** 3 - 3 * x[0] ** 2 - x[0]


def simulated_annealing(objective, bounds, iterations, step, temp):
    best = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
    best_eval = objective(best)
    curr, curr_eval = best, best_eval
    scores = list()

    for i in range(iterations):

        candidate = curr + randn(len(bounds)) * step
        candidate_eval = objective(candidate)

        if candidate_eval < best_eval:
            best, best_eval = candidate, candidate_eval
            scores.append(best_eval)
            print('>%d f(%s) = %.5f' % (i, best, best_eval))

        diff = candidate_eval - curr_eval
        t = temp / float(i + 1)
        metropolis = exp(-diff / t)

        if diff < 0 or rand() < metropolis:
            curr, curr_eval = candidate, candidate_eval

    return [best, best_eval, scores]


if __name__ == '__main__':
    seed(1)
    bounds = asarray([[-1.5, 1.5]])
    n_iterations = 1000
    step_size = 0.2
    temp = 100
    best, score, scores = simulated_annealing(objective, bounds, n_iterations, step_size, temp)
    print('Done!')
    print('f(%s) = %f' % (best, score))
