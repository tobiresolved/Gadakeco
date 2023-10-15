import sys
import random
import datetime
import copy
import pickle
from neat.network import Network
class Population():

    def __init__(self, seed, size):
        """
        Erstellt eine neue Population mit der Groesse 'size' und wird zuerst fuer den uebergebenen seed trainiert.
        """
        self.seed = seed
        # Das Attribut generation_count wird von Gadakeco automatisch inkrementiert. 
        self.generation_count = 1

        # eindeutiger name name des Netzwerks (noch zu implementieren)
        current_time = datetime.datetime.now()
        self.name = current_time.strftime("%Y-%m-%d %H_%M_%S")
        self.current_generation = []

        for _ in range(size):
            network = Network()
            network.add_connection()
            self.current_generation.append(network)


    @staticmethod
    def load_from_file(filename):

        try:
            with open(filename, 'rb') as file:
                population = pickle.load(file)
            if isinstance(population, Population):
                print(f"Population loaded from {filename}")
                return population
            else:
                print(f"Invalid data in file {filename}")
                return None
        except Exception as e:
            print(f"Error loading population from {filename}: {e}")
            return None

    def save_to_file(self, filename):
        try:
            with open(filename, 'wb') as file:
                pickle.dump(self, file)
            print(f"Population saved to {filename}")
        except Exception as e:
            print(f"Error saving population to {filename}: {e}")
        
        """
        Speichert die komplette Population in die Datei mit dem Pfad filename.
        """
        print("called save_to_file")
        

    def create_next_generation(self):
        current_generation = self.current_generation

        sorted_generation = sorted(current_generation, key=lambda x: x.fitness, reverse=True)
        cutoff_index = int(0.1 * len(sorted_generation))
        top_10_percent = sorted_generation[:cutoff_index]
        new_90_percent = []
        next_generation = []

        next_generation.extend(top_10_percent)
        
        for _ in range(9):
            deepcopy_top_10 = copy.deepcopy(top_10_percent)
            new_90_percent.extend(deepcopy_top_10)

        for mutate_network in new_90_percent:
            random_number = random.random()
            
            if random_number <= 0.8:
                mutate_network.add_connection()

            else:
                mutate_network.add_hidden_neuron()

        next_generation.extend(new_90_percent)
        self.current_generation = next_generation
       
        
        
        

