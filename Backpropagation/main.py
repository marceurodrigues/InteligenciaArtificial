import numpy as np

def backprop(self, x, y):
    """
    Retorna uma tupla '(nabla_b, nabla_w)' representando o
    gradiente para a função de custo C_x. 'nabla_b' e 'nabla_w'
    são listas de camadas de matrizes numpy, semelhantes a
    'self.biases' e 'self.weights'.
    """
    nabla_b = [np.zeros(b.shape) for b in self.biases]
    nabla_w = [np.zeros(w.shape) for w in self.weights]

    #Feedforward
    activation = x

    # Lista para armazenar todas as ativações, camada por camada
    activations = [x]

    #Lista para armazenar todos os vetores z, camada por camada
    zs = []

    for b, w in zip(self.biases, self.weights):
        z = np.dot(w, activation) + b
        zs.append(z)
        activation = sigmoid(z)
        activations.append(activation)

    # Backward pass
    delta = self.cost_derivate(activations[-1], y) * sigmoid_prime(zs[-1])
    nabla_b[-1] = delta
    nabla_w[-1] = np.dot(delta, activations[-2].transpose())

    # Aqui, l = 1 significa a última camada de neurônios, l = 2 é a segunda e assim por diante...
    for l in range(2, self.num_layers):
        z = zs[-l]
        sp = sigmoid_prime(z)
        delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
        nabla_b[-l] = delta
        nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
    return (nabla_b, nabla_w)