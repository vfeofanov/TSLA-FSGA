import numpy as np
from itertools import repeat, combinations
from joblib import Parallel, delayed, effective_n_jobs
from selection_metrics import OOBFeatureSelectionMetric
from copy import deepcopy


def compute_feature_weights(models, candidates):
    num_features = candidates.shape[1]

    def form_full_vector_weights(model, candidate, num_features):
        w = np.zeros(num_features)
        w[candidate] = model.feature_importances_
        return w

    weights_matrix = np.array(list(map(form_full_vector_weights, models, candidates, repeat(num_features))))
    return weights_matrix


def check_zero(population):
    population = np.delete(population, np.where(np.all((population - 1) < 0, axis=1)), axis=0)
    return population


def compute_score(model, x, y, candidate):
    model.fit(x, y, candidate)
    model.compute_value(x, y, candidate)
    return model


def compute_fitness(x, y, population, metric, n_jobs):
    candidates = population.astype('bool')
    if n_jobs == -1:
        n_jobs = min(effective_n_jobs(n_jobs), len(candidates))
        print(n_jobs)
    models = list(map(lambda i: deepcopy(metric), range(len(candidates))))
    models = Parallel(n_jobs=n_jobs)(delayed(compute_score)(models[i], x, y, candidates[i]) for i in range(len(candidates)))
    scores = np.array(list(map(lambda model: model.value, models)))
    learners = list(map(lambda model: model.learner_, models))
    weights = compute_feature_weights(learners, candidates)
    return scores, weights


def check_relevance(X, y, fitness, weights, population, feat_removed, w_threshold, relevance_threshold, metric):
    if type(X) is list:
        x = X[0]
    else:
        x = X
    best_subset = population[np.argmax(fitness), :]
    # find features with small average weights
    avg_weights = weights.sum(axis=0) / weights.sum()
    feat_small_weights = np.where(avg_weights <= w_threshold)[0]
    feat_small_weights = feat_small_weights[np.logical_not(np.isin(feat_small_weights, feat_removed))]
    # remove from consideration features that didn't appear in population at all
    where_zeros = np.where(np.logical_not(population.sum(axis=0)))[0]
    feat_small_weights = np.setdiff1d(feat_small_weights, where_zeros)
    # if nothing is found, return immediately
    if feat_small_weights.size == 0:
        return feat_removed, population
    # retrieve feature values of small weight variables and from the best subset
    x_feat_small_weights = x[:, feat_small_weights]
    x_best_subset = x[:, best_subset.astype('bool')]
    # random value permutations of small weight variables 
    x_artif_var = np.array(list(map(lambda column: np.random.choice(column, column.size, replace=False),
                                    x_feat_small_weights.T))).T
    # learn a joint model
    # random state should already be integrated into the learner
    if metric.learner_.__class__.__name__ == 'SelfLearning':
        learner = deepcopy(metric.learner_.init_model)
    else:
        learner = deepcopy(metric.learner_)
    learner.fit(np.hstack((x_best_subset, x_feat_small_weights, x_artif_var)), y)
    num_feats_best_subset = int(best_subset.sum())
    # retrieve feature weights
    fi1 = learner.feature_importances_[num_feats_best_subset:(num_feats_best_subset + feat_small_weights.size)]
    fi2 = learner.feature_importances_[(num_feats_best_subset + feat_small_weights.size):]
    # if the difference between the weights of a feature and its noisy counterpart is small,
    # we treat the variable as irrelevant
    irrelevant_feats = feat_small_weights[np.abs(fi1 - fi2) < relevance_threshold]
    feat_removed += irrelevant_feats.tolist()
    if len(feat_removed) != 0:
        population[:, np.array(feat_removed)] = 0
        weights[:, np.array(feat_removed)] = 0
    return feat_removed, population


def backfilling(parents, parent_weights, feat_removed, size):
    parent_sizes = parents.sum(axis=1)
    quotas = size - parent_sizes
    for i in range(parents.shape[0]):
        where_zeros = np.setdiff1d(np.where(np.logical_not(parents[i, :]))[0], feat_removed)
        if quotas[i] != 0:
            if where_zeros.size < quotas[i]:
                quotas[i] = where_zeros.size
            fillings = np.random.choice(where_zeros, quotas[i], replace=False)
            parents[i, fillings] = 1
            parent_weights[i, fillings] = 1e-10
    return parents, parent_weights


def parent_selection(population, fitness, num_parents, weights, feat_removed):
    # select feature subsets with largest fitness in population
    parents_ind = np.argsort(fitness)[::-1][:num_parents]
    parents = population[parents_ind, :]
    parent_weights = weights[parents_ind, :]
    parent_features = np.unique(np.where(parents)[1]).tolist()
    out_of_parent_features = np.delete(np.arange(population.shape[1]), parent_features + feat_removed)
    return parents, parent_weights, parent_features, out_of_parent_features


def mate_two_parents_standard(parents, parent_combination, proportion):
    num_weights = parents.shape[1]
    child = np.zeros(num_weights)
    parent_0 = parents[parent_combination[0], :]
    parent_1 = parents[parent_combination[1], :]
    child_length = parent_0.sum()
    crossover_point = np.uint8(child_length * proportion)
    where_ones_parent_0 = np.where(parent_0)[0]
    where_ones_parent_1 = np.where(parent_1)[0]
    taken_idx_parent_0 = np.random.choice(where_ones_parent_0, crossover_point, replace=False)
    child[taken_idx_parent_0] = 1
    # keep only the one that were not taken from parent 0
    where_ones_parent_1 = np.setdiff1d(where_ones_parent_1, taken_idx_parent_0)
    taken_idx_parent_1 = np.random.choice(where_ones_parent_1, child_length - crossover_point, replace=False)
    child[taken_idx_parent_1] = 1
    return child


def mate_two_parents_weighted(parent_weights, parent_combination, proportion, dominant):
    child = np.zeros(parent_weights.shape[1])
    parent_0_weights = parent_weights[parent_combination[0], :]
    parent_1_weights = parent_weights[parent_combination[1], :]
    if dominant == 0:
        child_length = np.sum(parent_0_weights != 0)
    else:
        child_length = np.sum(parent_1_weights != 0)
    crossover_point = np.uint8(child_length * proportion)
    taken_idx_parent_0 = np.argsort(parent_0_weights)[::-1][:crossover_point]
    child[taken_idx_parent_0] = 1
    sorted_idx_parent_1 = np.argsort(parent_1_weights)[::-1]
    taken_idx_parent_1 = sorted_idx_parent_1[np.logical_not(np.isin(sorted_idx_parent_1, taken_idx_parent_0))]
    # if the number of features taken from the parent 1 is less than its quota,
    # we just take all of them
    # otherwise, first [child_length - crossover_point] features.
    if taken_idx_parent_1.size < child_length - crossover_point:
        if taken_idx_parent_1.size != 0:
            child[taken_idx_parent_1] = 1
        # NEW: adding some features from parent 0
        rest_quota = child_length - crossover_point - taken_idx_parent_1.size
        taken_idx = np.argsort(parent_0_weights)[::-1][crossover_point:(crossover_point+rest_quota)]
        child[taken_idx] = 1
    else:
        child[taken_idx_parent_1[:(child_length - crossover_point)]] = 1
    return child


def crossover(parents, parent_weights, num_candidates, rand_cross_point=True, mate_type='weighted'):
    num_parents = parent_weights.shape[0]
    offspring_size = num_candidates - num_parents
    # all possible combinations between parents
    possible_parent_combinations = list(combinations(np.arange(num_parents), 2))

    if rand_cross_point:
        # proportions is an array of crossover point positions for offspring
        proportions = np.random.choice([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], offspring_size)
        # we take each time randomly one of the combinations
        ind_combinations = np.random.choice(np.arange(len(possible_parent_combinations)), offspring_size)
    else:
        # when it's not random, the center is taken
        proportions = np.repeat(0.5, offspring_size)
        # we take each time randomly one of the combinations
        ind_combinations = np.random.choice(np.arange(len(possible_parent_combinations)), offspring_size, replace=False)

    parent_combinations = np.array(possible_parent_combinations)[ind_combinations]
    dominants = np.random.choice([0, 1], offspring_size)
    if mate_type == 'weighted':
        offspring = np.array(
            list(map(mate_two_parents_weighted, repeat(parent_weights), parent_combinations, proportions, dominants)))
    if mate_type == 'standard':
        offspring = np.array(
            list(map(mate_two_parents_standard, repeat(parents), parent_combinations, proportions)))
    return offspring


def mutation(child, feat_removed, prob_mut, max_num_mut):
    # indices of child features (ones) and other features that are still under consideration (zeros)
    where_zeros = np.where(np.logical_not(child))[0]
    where_zeros = np.setdiff1d(where_zeros, feat_removed)
    where_ones = np.where(child)[0]
    # 1-prob_mut is the probability to not mutate features
    if max_num_mut == 0:
        return child
    if np.random.uniform(0, 1) > prob_mut:
        return child
    mut_num = np.random.choice(np.arange(max_num_mut) + 1, 1)
    if mut_num > where_zeros.size:
        mut_num = where_zeros.size
    if mut_num > where_ones.size:
        mut_num = where_ones.size
    chosen_vars_obt = np.random.choice(where_zeros, mut_num, replace=False)
    chosen_vars_rem = np.random.choice(where_ones, mut_num, replace=False)
    child[chosen_vars_obt] = 1
    child[chosen_vars_rem] = 0
    return child


def fsga(x, y, num_generations=10, num_candidates=40, num_parents=8, rand_cross_point=True, prob_mut=0.9,
         max_num_mut='1-2', w_threshold='1-5d', metric=None, random_state=None, n_jobs=4, report=0, relevance_test=True,
         mate_type='weighted', relevance_threshold=1e-04, n_feat=None):
    if metric is None:
        metric = OOBFeatureSelectionMetric(random_state=random_state)

    if report != 0:
        report_text = ""
    else:
        report_text = None

    # fix random seed
    np.random.seed(random_state)

    # number of total features
    if type(x) is list:
        n_all_feat = x[0].shape[1]
    else:
        n_all_feat = x.shape[1]
    # feature indices
    features = np.arange(n_all_feat)
    # features that are not considered in feature
    # selection anymore
    feat_removed = []
    # number of features we start from
    if n_feat is None:
        n_feat_ = int(np.sqrt(n_all_feat))
    else:
        n_feat_ = n_feat
    # heuristics for max num of mutations
    # default is 1/2 of the selected num of features
    if max_num_mut == '1-2' or max_num_mut is None:
        max_num_mut = int(n_feat_ / 2)
    # other heuristics
    if max_num_mut == '2-3':
        max_num_mut = int(n_feat_ / 1.5)
    if max_num_mut == '1-3':
        max_num_mut = int(n_feat_ / 3)

    # heuristics to choose w_threshold
    if w_threshold == '1-d':
        w_threshold = 1 / n_feat_
    if w_threshold == '1-5d':
        w_threshold = 1 / (5 * n_feat_)

    # number of possible mating between two parents
    num_mating = num_parents * (num_parents - 1) / 2
    if not rand_cross_point:
        if num_candidates - num_parents > num_mating:
            num_candidates = int(num_parents + num_mating)
            print("num_candidates was reduced to", num_candidates)

    features_for_init = features
    # creating the initial population.:
    # num_candidates candidates where each candidate has n_feat_ genes
    population_ind = list(map(lambda i: np.random.choice(features_for_init, size=n_feat_, replace=False),
                              np.arange(num_candidates)))

    def form_boolean_candidate(candidate):
        w = np.zeros(n_all_feat)
        w[candidate] = 1
        # return w!=0
        return w.astype('int')

    population = np.array(list(map(form_boolean_candidate, population_ind)))
    population = check_zero(population)
    num_candidates = population.shape[0]

    for generation in range(num_generations):
        if report > 0:
            report_text += "Generation: {}\n".format(generation)

        # measuring the fitness of each candidate in the population
        fitness, weights = compute_fitness(x, y, population, metric, n_jobs)

        # the best result in the current iteration.
        if report > 0:
            report_text += "Best fitness: {}\n".format(np.max(fitness))

        # selecting the best parents in the population for mating
        parents, parent_weights, parent_features, out_of_parent_features = parent_selection(population, fitness,
                                                                                            num_parents, weights,
                                                                                            feat_removed)
        if report > 1:
            report_text += "Parent Features: {}\n".format(parent_features)
            report_text += "Out of parent features: {}\n".format(out_of_parent_features)
            report_text += "Parents: \n{}\n".format(parents)

        if relevance_test:
            # removing features from consideration with small weights and irrelevant to the response
            feat_removed, population = check_relevance(x, y, fitness, weights, population, feat_removed,
                                                       w_threshold, relevance_threshold, metric)
            parents, parent_weights = backfilling(parents, parent_weights, feat_removed, n_feat_)
            if report > 1:
                report_text += "Removed features: {}\n".format(feat_removed)

        # offspring
        # generating next generation using crossover.
        offspring_crossover = crossover(parents, parent_weights, num_candidates, rand_cross_point, mate_type)

        # offspring mutation
        offspring_mutation = np.array(list(map(mutation, offspring_crossover, repeat(feat_removed), repeat(prob_mut),
                                               repeat(max_num_mut))))

        if report > 1:
            report_text += "Offspring: \n{}\n".format(offspring_mutation)

        # creating the new population based on the parents and offspring.
        population[:num_parents, :] = parents
        population[num_parents:, :] = offspring_mutation
        population = check_zero(population)
        num_candidates = population.shape[0]

    #  final population
    # at first, the fitness is calculated for each solution in the final generation.
    fitness, weights = compute_fitness(x, y, population, metric, n_jobs)

    # then return the index of that solution corresponding to the best fitness.
    best_match_idx = np.argmax(fitness)

    if report > 0:
        report_text += "Best solution: {}\n".format(population[best_match_idx, :])
        report_text += "Best solution fitness: {}\n".format(fitness[best_match_idx])
        report_text += "Num of removed feats: {}\n".format(len(feat_removed))
        report_text += "Num of parent feats: {}\n".format(len(parent_features))
        report_text += "Num of out of parent feats: {}\n".format(len(out_of_parent_features))

    return fitness, population, best_match_idx, weights, len(feat_removed), report_text


class FeatureSelectionGeneticAlgorithm:

    def __init__(self, num_generations=20, num_candidates=40, num_parents=8, rand_cross_point=True, prob_mut=1,
                 max_num_mut='1-2', w_threshold='1-5d', metric=None, relevance_test=True, mate_type='weighted',
                 relevance_threshold=1e-04, n_feat=None, random_state=None, n_jobs=4, report=0):

        self.num_generations = num_generations
        self.num_candidates = num_candidates
        self.num_parents = num_parents
        self.rand_cross_point = rand_cross_point
        self.prob_mut = prob_mut
        self.max_num_mut = max_num_mut
        self.w_threshold = w_threshold
        self.metric = metric
        self.relevance_test = relevance_test
        self.mate_type = mate_type
        self.relevance_threshold = relevance_threshold
        self.n_feat = n_feat
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.report = report
        self.fitness = None
        self.population = None
        self.best_match_idx = None
        self.weights = None
        self.report_text = None
        self.subset = None
        self.n_feat_removed = None

    def fit(self, x, y):
        params = {
            'x': x,
            'y': y,
            'num_generations': self.num_generations,
            'num_candidates': self.num_candidates,
            'num_parents': self.num_parents,
            'rand_cross_point': self.rand_cross_point,
            'prob_mut': self.prob_mut,
            'max_num_mut': self.max_num_mut,
            'w_threshold': self.w_threshold,
            'metric': self.metric,
            'random_state': self.random_state,
            'n_jobs': self.n_jobs,
            'report': self.report,
            'relevance_test': self.relevance_test,
            'mate_type': self.mate_type,
            'relevance_threshold': self.relevance_threshold,
            'n_feat': self.n_feat,
        }

        fitness, population, best_match_idx, weights, n_feat_removed, report_text = fsga(** params)
        self.fitness = fitness
        self.population = population
        self.best_match_idx = best_match_idx
        self.weights = weights
        self.n_feat_removed = n_feat_removed
        if report_text is not None:
            self.report_text = report_text
        self.subset = population[best_match_idx, :].astype('bool')

    def select_features(self, n_feat):
        return self.subset
