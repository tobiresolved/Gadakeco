import time
import numpy as np
import random
import matplotlib.pyplot as plt

class Neuron:       
    def __str__(self):
        return "Neuron: " + self.name + " OutputValue: " + str(self.output_value)
        
    
    def calculate_output_value(self):
            sum = 0
            for neuron_tuple in self.connections:
                prior_neuron_value = neuron_tuple[0].output_value
                
                weight = neuron_tuple[1]
                if weight != None:
                    value =  prior_neuron_value * weight
                else:
                    value = 0
                sum = sum + value

        
            self.output_value = np.sign(sum)     
        
class InputNeuron(Neuron):
    def __init__(self, id, pixel_input_value = 0):
        self.id = id
        self.name = "I_" + str(self.id)
        self.output_value = pixel_input_value

class OutputNeuron(Neuron):
    def __init__(self, id):
        self.connections = []
        self.output_value = None
        self.id = id
        self.name = "O_" + str(self.id)

        


class HiddenNeuron(Neuron):
    def __init__(self, id, layer):
        self.connections = []
        self.output_value = None
        self.id = id
        self.layer = layer
        self.name = str(self.layer) + "_H_" + str(self.id)
        self.x = None
        self.y = None
        
 
class Network:
    def __init__(self):
        self.hidden_list = []
        self.createIO_layers(486, 3)
        self.connection_list = []
        self.fitness = 0
    
    def createIO_layers(self, input_size, output_size):
        self.input_list = []
        self.output_list = []
        for number in range(1, (input_size+1)):
            self.input_list.append(InputNeuron(number))
        for number in range(1,(output_size+1)):
            self.output_list.append(OutputNeuron(number))

    def update_fitness(self, points, time):
        """
        Berechnet und aktualisiert den Fitness-Wert des Netzwerks 
        basierend auf den Punkten (des 'Spielers') und der vergangenen Zeit.
        """
        self.fitness = np.round(points - 50 * time)

    def evaluate(self, values):
        """
        Wertet das Netzwerk aus. 
        Argumente:
            values: eine Liste von 27x18 = 486 Werten, welche die aktuelle diskrete Spielsituation darstellen
                    die Werte haben folgende Bedeutung:
                     1 steht fuer begehbaren Block
                    -1 steht fuer einen Gegner
                     0 leerer Raum
        Rueckgabewert:
            Eine Liste [a, b, c] aus 3 Boolean, welche angeben:
                a, ob die Taste "nach Links" gedrueckt ist
                b, ob die Taste "nach Rechts" gedrueckt ist
                c, ob die Taste "springen" gedrueckt ist.
        """
        for i in range(len(values)):
            self.input_list[i].output_value = values[i]

        self.calculate_layerwise_output()

        left = self.output_list[0].output_value
        right = self.output_list[1].output_value
        jump = self.output_list[2].output_value

        press_left = left > 0
        press_right = right > 0
        press_jump = jump > 0

        return [press_left, press_right, press_jump]

    def get_gaussian_pixel(self):
        mean_x = 10  
        std_dev_x =  np.sqrt(27)
        mean_y = 10 
        std_dev_y =  np.sqrt(18)

        random_numbers_x = np.random.normal(mean_x, std_dev_x, size=1)
        random_numbers_y = np.random.normal(mean_y, std_dev_y, size=1)

        random_numbers_x = np.clip(random_numbers_x, 1, 27)
        random_numbers_y = np.clip(random_numbers_y, 1, 18)

        random_numbers_x = np.round(random_numbers_x).astype(int)
        random_numbers_y = np.round(random_numbers_y).astype(int)

        pixel = (random_numbers_y - 1) * 27 + random_numbers_x
        pixel_neuron = "I_" + str(pixel[0])
            
        for InputNeuron in self.input_list:
            if InputNeuron.name == pixel_neuron:
                return InputNeuron

        
    def get_from_connection_neuron(self):
        input_pixel = self.get_gaussian_pixel()
        if len(self.hidden_list) > 0:
            random_element = random.choice(self.hidden_list)
            chosen_element = random.choices([input_pixel,random_element], weights=[0.75, 0.25], k=1)
            return chosen_element[0]
        else:
            return input_pixel

    def get_to_connection_neuron(self, from_neuron):
        possible_neurons = []
        possible_neurons.extend(self.output_list)
        if isinstance(from_neuron,InputNeuron):
            for hidden_neuron in self.hidden_list:
                if hidden_neuron.layer > 0:
                    possible_neurons.append(hidden_neuron)
            to_neuron = random.choice(possible_neurons)
            return to_neuron

        elif isinstance(from_neuron,HiddenNeuron):
            for hidden_neuron in self.hidden_list:
                if hidden_neuron.layer > from_neuron.layer:
                    possible_neurons.append(hidden_neuron)
            to_neuron = random.choice(possible_neurons)
            return to_neuron
        
    def add_in_connection(self, from_neuron, to_neuron, weight = 1):
        connection_added = (from_neuron, to_neuron)
        
        if not connection_added in self.connection_list:
                        self.connection_list.append(connection_added)
                        to_neuron.connections.append((from_neuron, weight))
                        self.clean_hidden_layers()
                        

    def add_connection(self):
        from_neuron = self.get_from_connection_neuron()
        to_neuron = self.get_to_connection_neuron(from_neuron)
        weight = random.choice([1,-1])
        connection_added = (from_neuron, to_neuron)
        if not connection_added in self.connection_list:
            self.connection_list.append(connection_added)
            to_neuron.connections.append((from_neuron, weight))



        
   
        
    def add_hidden_neuron(self):
        if len(self.connection_list) == 0:
            print(self.connection_list)
            print("Connection LIST IS EMPTY")
        else:
            connection = random.choice(self.connection_list)
            self.connection_list.remove(connection)
            from_neuron = connection[0]
            to_neuron = connection[1]
            found_weight = None
            for target_neuron, weight in to_neuron.connections:
                if target_neuron == from_neuron:
                    found_weight = weight
                    to_neuron.connections.remove((from_neuron, weight))
                break
            weight = found_weight
            hidden_neuron_id = len(self.hidden_list) + 1
            hidden_neuron_layer = 0
            if isinstance(from_neuron, InputNeuron) and isinstance(to_neuron, OutputNeuron):
                hidden_neuron_layer = 1
            elif isinstance(from_neuron, HiddenNeuron):
                hidden_neuron_layer = from_neuron.layer + 1
            new_hidden_neuron = HiddenNeuron(hidden_neuron_id,hidden_neuron_layer)
            self.hidden_list.append(new_hidden_neuron)
            self.add_in_connection(from_neuron=from_neuron, to_neuron=new_hidden_neuron)
            self.add_in_connection(from_neuron = new_hidden_neuron,to_neuron = to_neuron, weight=weight)
            to_neuron.output_value = 0

    def clean_hidden_layers(self):
        for connection in self.connection_list:
            if (isinstance(connection[0], HiddenNeuron) and isinstance(connection[1], HiddenNeuron)):
                if connection[0].layer == connection[1].layer:
                    connection[1].layer = connection[1].layer + 1
                    self.clean_hidden_layers()
                

    def calculate_layerwise_output(self):
        sorted_hidden_layer = sorted(self.hidden_list, key=lambda x: x.layer)
        for hidden_neuron in sorted_hidden_layer:
            hidden_neuron.calculate_output_value()
        for output_neuron in self.output_list:
            output_neuron.calculate_output_value()
