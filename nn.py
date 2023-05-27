import random
import math

alpha = 0.2

def make(nx, nz, ny):
    """Funcao que cria, inicializa e devolve uma rede neuronal, incluindo
    a criacao das diversos listas, bem como a inicializacao das listas de pesos. 
    Note-se que sao incluidas duas unidades extra, uma de entrada e outra escondida, 
    mais os respectivos pesos, para lidar com os tresholds; note-se tambem que, 
    tal como foi discutido na teorica, as saidas destas estas unidades estao sempre a -1.
    por exemplo, a chamada make(3, 5, 2) cria e devolve uma rede 3x5x2"""
   
    """A function that creates, initializes, and returns a neural network, including
    the creation of the various lists as well as the initialization of the lists of weights. 
    Note that two extra units are included, one input and one hidden, plus their respective weights, to handle the tresholds; 
    also note that, as discussed in the theorem, the outputs of these units are always at -1. 
    For instance, the call make(3, 5, 2) creates and returns a 3x5x2 network"""
    
    
    #a rede neuronal é num dicionario com as seguintes chaves:
    # nx     numero de entradas
    # nz     numero de unidades escondidas
    # ny     numero de saidas
    # x      lista de armazenamento dos valores de entrada
    # z      array de armazenamento dos valores de activacao das unidades escondidas
    # y      array de armazenamento dos valores de activacao das saidas
    # wzx    array de pesos entre a camada de entrada e a camada escondida
    # wyz    array de pesos entre a camada escondida e a camada de saida
    # dz     array de erros das unidades escondidas
    # dy     array de erros das unidades de saida    
    
    nn = {'nx':nx, 'nz':nz, 'ny':ny, 'x':[], 'z':[], 'y':[], 'wzx':[], 'wyz':[], 'dz':[], 'dy':[]}
    
    nn['wzx'] = [[random.uniform(-0.5,0.5) for _ in range(nn['nx'] + 1)] for _ in range(nn['nz'])]
    nn['wyz'] = [[random.uniform(-0.5,0.5) for _ in range(nn['nz'] + 1)] for _ in range(nn['ny'])]
    return nn

def sig(input):
    """Funcao de activacao (sigmoide)"""
    """Activation function (sigmoidal)"""
    
    return 1.0/(1.0 + math.exp(- input))


def forward(nn, input):
    nn['x']=input.copy()
    nn['x'].append(-1)
    
    #calcula a activacao da unidades escondidas
    nn['z']=[sig(sum([x*w for x, w in zip(nn['x'], nn['wzx'][i])])) for i in range(nn['nz'])]
    nn['z'].append(-1)
    
    #calcula a activacao da unidades de saida
    nn['y']=[sig(sum([z*w for z, w in zip(nn['z'], nn['wyz'][i])])) for i in range(nn['ny'])]
 
   
def error(nn, output):
    """Funcao que recebe uma rede nn com as activacoes calculadas
       e a lista output de saidas pretendidas e calcula os erros
       na camada escondida e na camada de saida"""
    """Function that receives a net nn with the calculated activations 
    and the output list of desired outputs and calculates the errors
    in the hidden layer and the output layer"""
    
    nn['dy']=[y*(1-y)*(o-y) for y,o in zip(nn['y'], output)]
    
    zerror=[sum([nn['wyz'][i][j]*nn['dy'][i] for i in range(nn['ny'])]) for j in range(nn['nz'])]
    
    nn['dz']=[z*(1-z)*e for z, e in zip(nn['z'], zerror)]
 
 
def update(nn):
    """funcao que recebe uma rede com as activacoes e erros calculados e
    actualiza as listas de pesos"""
    """function that receives a net with the calculated activations and errors 
    and updates the weight lists"""
    
    nn['wzx'] = [[w+x*nn['dz'][i]*alpha for w, x in zip(nn['wzx'][i], nn['x'])] for i in range(nn['nz'])]
    nn['wyz'] = [[w+z*nn['dy'][i]*alpha for w, z in zip(nn['wyz'][i], nn['z'])] for i in range(nn['ny'])]
    

def iterate(i, nn, input, output):
    """Funcao que realiza uma iteracao de treino para um dado padrao de entrada input
    com saida desejada output"""
    """Function that performs a training operation for a given input pattern 
    with desired output"""
    
    forward(nn, input)
    error(nn, output)
    update(nn)
    print('%03i: %s -----> %s : %s' %(i, input, output, nn['y']))
    


def train_and():
    """Funcao que cria uma rede 2x2x1 e treina um AND"""
    
    net = make(2, 2, 1)
    for i in range(2000):
        iterate(i, net, [0, 0], [0])
        iterate(i, net, [0, 1], [0])
        iterate(i, net, [1, 0], [0])
        iterate(i, net, [1, 1], [1])
    
def train_or():
    """Funcao que cria uma rede 2x2x1 e treina um OR"""
    
    net = make(2, 2, 1)
    for i in range(1000):
        iterate(i, net, [0, 0], [0])
        iterate(i, net, [0, 1], [1])
        iterate(i, net, [1, 0], [1])
        iterate(i, net, [1, 1], [1]) 

def train_xor():
    """Funcao que cria uma rede 2x2x1 e treina um XOR"""
    
    net = make(2, 2, 1)
    for i in range(10000):
        iterate(i, net, [0, 0], [0])
        iterate(i, net, [0, 1], [1])
        iterate(i, net, [1, 0], [1])
        iterate(i, net, [1, 1], [0]) 
    


def run():
    """Funcao principal do nosso programa, cria os conjuntos de treino e teste, chama
    a funcao que cria e treina a rede e, por fim, a funcao que a treina"""
    """Main function of our program, it creates the training and test sets, calls
     the function that creates and trains the network and, finally, the function that trains it"""
    
    training_set, test_set = build_sets("zoo.txt")
    nn= train_zoo(training_set)
    test_zoo(nn, test_set)
    


def build_sets(f):
    """Funcao que cria os conjuntos de treino e de de teste a partir dos dados
    armazenados em f (zoo.txt). A funcao le cada linha, tranforma-a numa lista
    de valores e chama a funcao translate para a colocar no formato adequado para
    o padrao de treino. Estes padroes são colocados numa lista 
    Finalmente, devolve duas listas, uma com os primeiros 67 padroes (conjunto de treino)
    e a segunda com os restantes (conjunto de teste)"""
    file1 = open('zoo.txt', 'r')
    Lines = file1.readlines()
    compt=0
    training_set=[]
    test_set= []
    shuf = []
    for line in Lines:
        line = line[1:-2]

        # Split the line by commas
        items = line.split(",")

        # Convert each item to the appropriate data type
        items = [int(item) if item.isdigit() else item for item in items]
        print(items)  
        shuf.append(items)    
        # random.shuffle(translate(items))
    #print(zeb)
    random.shuffle(shuf)
    for i in shuf:
        compt+=1
        if compt<=67:       
            training_set.append(translate(i))
        else:
            test_set.append(translate(i))        
    return training_set,test_set


def translate(lista):
    """Recebe cada lista de valores e transforma-a num padrao de treino.
    Cada padrao tem o formato [nome_do_animal, padrao_de_entrada, tipo_do_animal, padrao_de_saida].
    nome_do_animal e o primeiro valor da lista e tipo_de_animal o ultimo.
    padrao_de_entrada e uma lista de 0 e 1 com os valores dos atributos.
    O numero de pernas deve tambem ser convertido numa lista de 0 e 1, concatenada com os restantes
    atributos. E.g. [0 0 0 0 1 0 0 0 0 0] -> 4 pernas.
    padrao_de_saida e uma lista de 0 e 1 que representa o tipo do animal. Tem 7 posicoes e a unica
    que estiver a 1 corresponde ao tipo do animal. E.g., [0 0 1 0 0 0 0] -> reptile.
    """
    
    first_element = lista[0]
    last_element = lista[-1]
    attributes_elements = lista[1:-1]
    final_list = []
    final_list.append(first_element)
    final_list.append(attributes_elements)
    final_list.append(last_element)
    
    # Adding the type list
    attribute_type = ["mammal", "bird", "reptile", "fish", "amphibian", "insect", "invertebrate"]
    list_type = [0, 0, 0, 0, 0, 0, 0]
    index_type = attribute_type.index(final_list[2])
    list_type[index_type] = 1
    
    # Adding the legs list
    attribute_legs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    list_legs = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    list = final_list[1]
    # Separating the first and last parts of the list
    first_part = list[:12]
    second_part = list[13:]
    i = list[12]
    index_legs = attribute_legs.index(i)
    list_legs[index_legs] = 1
    
    final_list2 = []
    final_list2 = first_part + list_legs + second_part
    
    final_list[1] = final_list2
    final_list.append(list_type)
    return final_list
        

def train_zoo(training_set):
    """cria a rede e chama a funçao iterate para a treinar. Use 300 iteracoes"""
    #create the network and call the iterate function to train it. Use 300 iterations
    #def iterate(i, nn, input, output):
    net = make(25, 13, 7)

    for j in range (300):
        for i in range(len(training_set)):
            
            iterate(j,net,training_set[i][1],training_set[i][3])
    return net
   
        
    

def retranslate(out):
    """recebe o padrao de saida da rede e devolve o tipo de animal corresponte.
    Devolve o tipo de animal corresponde ao indice da saida com maior valor."""
    """receives the output pattern from the network and returns the corresponding animal type.
     Returns the type of animal corresponding to the output index with the highest value."""
    animal_type=[ 'mammal' , 'bird' , 'reptile' , 'fish' , 'amphibian' , 'insect',  'invertebrate']
    return animal_type[out]
    
    

def test_zoo(net, test_set):
    """Funcao que avalia a precisao da rede treinada, utilizando o conjunto de teste.
    Para cada padrao do conjunto de teste chama a funcao forward e determina o tipo
    do animal que corresponde ao maior valor da lista de saida. O tipo determinado
    pela rede deve ser comparado com o tipo real, sendo contabilizado o número
    de respostas corretas. A função calcula a percentagem de respostas corretas"""
    """Function that evaluates the accuracy of the trained network, using the test set.
     For each pattern in the test set, call the forward function and determine the type of animal that corresponds to the largest value in the output list. The type determined by the network must be compared with the real type, counting the number of correct answers. The function calculates the percentage of correct answers"""
    liste=[]
    for i in range(len(test_set)):
        forward(net,test_set[i][1])
        liste.append(net['y'])
    success_rate = 0
    for l in range(len(liste)) :
        for li in liste[l]:
            x=max(liste[l])
            indx=liste[l].index(x)
        print("The network thinks ", test_set[l][0] ," is a ",retranslate(indx)," it should be a ", test_set[l][2])
        if test_set[l][2] == retranslate(indx):
            success_rate += 1
    print((success_rate * 100) / len(test_set))
      
    
    

    


if __name__ == "__main__":
    #train_and()
   
    
    run()
    