import pygame
from neat.network import HiddenNeuron
from neat.network import OutputNeuron
from neat.network import InputNeuron
from collections import Counter




TILESIZE = 10


def render_network(surface, network, values):
    sf = surface
    nw = network
    """
    Zeichnet die Minimap und den Netzwerkgraphen 
    
    Argumente:
        surface: ein pygame.Surface der Groesse 750 x 180 Pixel.
                 Darauf soll der Graph und die Minimap gezeichnet werden.
        network: das eigen implementierte Netzwerk (in network.py), dessen Graph gezeichnet werden soll.
        values: eine Liste von 27x18 = 486 Werten, welche die aktuelle diskrete Spielsituation darstellen
                die Werte haben folgende Bedeutung:
                 1 steht fuer begehbaren Block
                -1 steht fuer einen Gegner
                 0 leerer Raum
                Die Spielfigur befindet sich immer ca. bei der Position (10, 9) und (10, 10).
    """
    colors = {1: (255, 255, 255), -1: (255, 0, 0)}
    # draw slightly gray background for the minimap
    pygame.draw.rect(surface, (128, 128, 128, 128), (0, 0, 27 * TILESIZE, 18 * TILESIZE))
    
    # draw minimap
    for y in range(18):
        for x in range(27):
            if values[y * 27 + x] != 0:
                color = colors[values[y * 27 + x]]
                surface.fill(color, (TILESIZE * x, TILESIZE * y, TILESIZE, TILESIZE))
                pygame.draw.rect(surface, (0, 0, 0), (TILESIZE * x, TILESIZE * y, TILESIZE, TILESIZE), 1)
  

    draw_output_neurons(surface=sf, network=nw)
    draw_input_neurons(surface=sf, network=nw)
    draw_hidden_neurons(surface=sf, network=nw)
    draw_connections(surface=sf,network=nw)

def draw_input_neurons(surface,network):
    network = network
    surface = surface

    pygame.draw.rect(surface, (100, 100, 100), (TILESIZE * 10, TILESIZE * 10, TILESIZE, TILESIZE), 3)
    pygame.draw.rect(surface, (100, 100, 100), (TILESIZE * 10, TILESIZE * 9, TILESIZE, TILESIZE), 3)
    for connections in network.connection_list:
        from_neuron = connections[0]
        to_neuron = connections[1]
        if isinstance(from_neuron, InputNeuron):
            y = from_neuron.id//27
            x = from_neuron.id%27
            if from_neuron.output_value > 0:
                pygame.draw.rect(surface, (0, 255, 0), (TILESIZE * x, TILESIZE * y, TILESIZE, TILESIZE), 1)
            elif from_neuron.output_value == -1:
                pygame.draw.rect(surface, (255, 0, 0), (TILESIZE * x, TILESIZE * y, TILESIZE, TILESIZE), 1)
            else:
                pygame.draw.rect(surface, (0, 0, 0), (TILESIZE * x, TILESIZE * y, TILESIZE, TILESIZE), 1)

def draw_output_neurons(surface,network):
    text_list = ["left", "right", "jump"]
    font = pygame.font.Font(None, 24)
    text_center = 15
    for text in text_list:

        draw_text = font.render(text, True, (0, 0, 0))
        text_rect = draw_text.get_rect()
        text_rect.left = 710
        text_rect.centery = text_center
        surface.blit(draw_text, text_rect)
        text_center = text_center + 75

    y = 15
    x = 685

    for output_neuron in network.output_list:
        pygame.draw.circle(surface, (0,0,0), (x, y), 15)

        if output_neuron.output_value == 1:
            pygame.draw.circle(surface, (0,255,0), (x, y), 12)

        else:
            pygame.draw.circle(surface, (255,255,255), (x, y), 12)
        y = y + 75

def draw_hidden_neurons(surface,network):

    layers = {}
    for neuron in network.hidden_list:
        
        layer_num = neuron.layer
        
        if layer_num not in layers:
            layers[layer_num] = []
        layers[layer_num].append(neuron)

    
 
    
    layer_sizes = {layer: len(neurons) for layer, neurons in layers.items()}
    if layer_sizes:

        max_layer = max(layer_sizes.keys())
        matrix_width = max_layer + 2
        matrix_height = max(layer_sizes.values()) + 1
        counter = 0

        cell_width = 386/matrix_width


        for layer, num_neurons in layer_sizes.items():
            for i in range(num_neurons):
                cell_height = 166/(num_neurons + 1)
                neuron = layers[layer][i]
                layer_neur = (neuron.layer + 1)
                neuron_number = i + 1
        

                neuron.x = 277 + (layer_neur * cell_width)
                neuron.y = 7 + (neuron_number * cell_height)
        

                pygame.draw.circle(surface, (0,0,0), (int(neuron.x), int(neuron.y)), 7)
                counter = counter + 1
                if neuron.output_value == 1:
                    pygame.draw.circle(surface, (0,255,0), (int(neuron.x), int(neuron.y)), 5)
                elif neuron.output_value == -1:
                    pygame.draw.circle(surface, (255,0,0), (int(neuron.x), int(neuron.y)), 5) 
                else:
                    pygame.draw.circle(surface, (255,255,255), (int(neuron.x), int(neuron.y)), 5)
                layer_neur = 0
                neuron_number = 0

        

    
def draw_connections(surface,network):

    for connections in network.connection_list:
        from_neuron = connections[0]
        to_neuron = connections[1]
        if isinstance(from_neuron, InputNeuron) and isinstance(to_neuron,OutputNeuron):
            draw_io_connections(surface=surface,network=network, input_neuron=from_neuron, output_neuron=to_neuron)
        else:
            draw_connection(surface=surface,network=network, input_neuron=from_neuron, output_neuron=to_neuron)
            
        

def draw_io_connections(surface,network, input_neuron , output_neuron):
        line_color = (0,0,0)
        for searched_neuron, weight in output_neuron.connections:
            if searched_neuron == input_neuron:
                if weight == 1:
                    line_color = (0,255,0)
                elif weight == -1:
                    line_color = (255,0,0)
        

        y = input_neuron.id//27 *10 + 5
        x = input_neuron.id%27 * 10 + 5
        if output_neuron.name == "O_1":
            o1_x = 685
            o1_y = 15
            pygame.draw.line(surface,line_color,(x,y),(o1_x,o1_y))
        if output_neuron.name == "O_2":
            o2_x = 685
            o2_y = 90
            pygame.draw.line(surface,line_color,(x,y),(o2_x,o2_y))
        if output_neuron.name == "O_3":
            o3_x = 685
            o3_y = 165
            pygame.draw.line(surface, line_color,(x,y),(o3_x,o3_y))

def draw_connection(surface,network, input_neuron, output_neuron):

    if isinstance(input_neuron, InputNeuron):
        y = input_neuron.id//27 *10 + 5
        x = input_neuron.id%27 * 10 + 5
    elif isinstance(input_neuron, HiddenNeuron):
        x = input_neuron.x
        y = input_neuron.y
    if isinstance(output_neuron, HiddenNeuron):
        o_x = output_neuron.x
        o_y = output_neuron.y
        pygame.draw.line(surface,(0,255,0),(x,y),(o_x,o_y))
        
    elif output_neuron.name == "O_1":
            o1_x = 685
            o1_y = 15
            pygame.draw.line(surface,(0,255,0),(x,y),(o1_x,o1_y))
    elif output_neuron.name == "O_2":
            o2_x = 685
            o2_y = 90
            pygame.draw.line(surface,(0,255,0),(x,y),(o2_x,o2_y))
    elif output_neuron.name == "O_3":
            o3_x = 685
            o3_y = 165
            pygame.draw.line(surface,(0,255,0),(x,y),(o3_x,o3_y))

    






        
        

    
        

    # Zeichnen Sie hier das Netzwerk auf das Surface.
