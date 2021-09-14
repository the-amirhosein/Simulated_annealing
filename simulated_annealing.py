from numpy import asarray
from numpy import exp
from numpy.random import randn
from numpy.random import rand


def objective(x):
    return -3 * x[0] ** 5 + 3 * x[0] ** 4 + 5 * x[0] ** 3 - 3 * x[0] ** 2 + x[0]


def simulated_annealing(objective, bounds, n_iterations, step_size, temp):
    best = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
    best_eval = objective(best)
    curr, curr_eval = best, best_eval
    scores = list()

    for i in range(n_iterations):

        candidate = curr + randn(len(bounds)) * step_size
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
    bounds = asarray([[-5.0, 5.0]])
    n_iterations = 1000
    step_size = 0.1
    temp = 10
    best, score, scores = simulated_annealing(objective, bounds, n_iterations, step_size, temp)
    print('Done!')
    print('f(%s) = %f' % (best, score))
