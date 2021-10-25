
import logging
from optimizer import Optimizer
import numpy as np
#from tqdm import tqdm

# Setup logging.
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.DEBUG,
    filename='log.txt'
)

def train_networks(networks, dataset):
   
    #pbar = tqdm(total=len(networks))
    for network in networks:
        network.train(dataset)
       # pbar.update(1)
    #pbar.close()


def get_max_accuracy(networks):
    total_accuracy = 0
    max_accuracy = 0
    min_accuracy = 0
    accuracy_list = []
    precision_list = []
    recall_list = []
    for network in networks:
        accuracy_list.append(network.accuracy)
        precision_list.append(network.precision)
        recall_list.append(network.recall)
        id = np.argmax(accuracy_list)
    # mean_acc, max_acc, min_acc, std_acc = np.mean(accuracy_list), np.max(accuracy_list), \
    #                                       np.min(accuracy_list), np.std(accuracy_list)
    # return mean_acc, max_acc, min_acc, std_acc
    # return total_accuracy / len(networks), max_accuracy, min_accuracy
    return accuracy_list[id], precision_list[id], recall_list[id]

def generate(generations, population, nn_param_choices, dataset):
   
    optimizer = Optimizer(nn_param_choices)
    networks = optimizer.create_population(population)

    # Evolve the generation.
    for i in range(generations):
        logging.info("***Doing generation %d of %d***" %
                     (i + 1, generations))

        # Train and get accuracy for networks.
        train_networks(networks, dataset)

        # Get the maximal accuracy, precision, recall for this generation.
        max_accuracy, max_precision, max_recall = get_max_accuracy(networks)

        # Print out the maximal accuracy, precision, recall each generation.
        logging.info("Generation max accuracy, precision, recall, max, min: %.4f%%, %.4f%%, %.4f%%" %
                     (max_accuracy, max_precision, max_recall))
        logging.info('-'*80)

        # Evolve, except on the last iteration.
        if i != generations - 1:
            # Do the evolution.
            networks = optimizer.evolve(networks)

    # Sort our final population.
    networks = sorted(networks, key=lambda x: x.accuracy, reverse=True)

    # Print out the top 5 networks.
    print_networks(networks[:5])

def print_networks(networks):
   
    logging.info('-'*80)
    for network in networks:
        network.print_network()

def main():
    """Evolve a network."""
    generations = 10  # Number of times to evole the population.
    population = 20  # Number of networks in each generation.
    dataset = 'cifar10'

    nn_param_choices = {
        # 'nb_layers': [ 1, 2, 3, 4],
        'learning_rate': [0.3, 0.25, 0.2, 0.15, 0.1, 0.01, 0.001, 0.0001],
        # 'weight_decay': [0.3, 0.1, 0.001, 0.01, 0.0001],
        'momentum': [0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
        # 'activation': ['relu'],
        
    }

    logging.info("***Evolving %d generations with population %d***" %
                 (generations, population))

    generate(generations, population, nn_param_choices, dataset)

if __name__ == '__main__':
    main()
