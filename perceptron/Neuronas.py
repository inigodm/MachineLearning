import numpy
import matplotlib.pyplot as plt
from Data import Data
from Drawer import Drawer
class Perceptron(object):
    def __init__(self, m=0.01, n_iter = 50, random_state =1):
        """
        Una neurona vendria a ser que, dados varos valores que consideramos que definen la salida, y asignandoles
        un peso, obtenemos la salida bajo la formula:
        z = W*X
        Siendo z la salida del sistema al asignar el peso W a las entradas X (son 2 vectores) de modo que 
        
        La decision sera f(z) = 1 sii z > 0 y -1 sii z<=0

        De modo que lo que hacemos para calcular el resultado mediante aprendizaje es:
        1- Iniciar a 0 o a numeros aleatorios pequenos los pesos (W)
        2- Para cada variable de cada entrenamiento (xi) actualizar el valor del peso: wi = wi + Awi siendo Awi lo 
            que tenemos que variar el peso que definimos por:
            
            Awi=m(y_real_i_j - y_predicha_i_j)x_i
        
            Siendo
            Awi            => Incremento del peso de la variable i (A-> incremento de)
            m              => Eta en el libro, el factor de aprendizaje
            y_predicha_i_j => La salida estimada con el peso actual para el entrenamiento j
            y_real_i_j     => la salida real con el peso actual para el entrenamiento j
            x_i            => variable i-esima de la medicion
        
        El parametro n_iter indica el numero de iteraciones sobre los datos de entrada que vamos a hacer para entrenar la neurona
        El parametro random_state indica un valor aleatorio de entrada para generar los primeros pesos
        """
        self.m = m
        self.n_iter = n_iter
        self.random_state = random_state

    def entrenar(self, X, y):
        """
        Siendo 
        X el array bidimensional de entrada  [x1_1, x2_1,... xn_1:x1_2, x2_2...xn_2:....]
        y el array de salida (un valor para cada conjunto de variables de entrada)
        """
        self.errors = []
        self.inicializarW(vector_size = 1+X.shape[1])
        for _ in range(self.n_iter):
            self.actualizaPesos(X, y)
        return self


    def inicializarW(self, vector_size):
        """
        Iniciar a 0 o a numeros aleatorios pequenos los pesos (W)
        """
        rgen = numpy.random.RandomState(self.random_state)
        self.w = rgen.normal(loc=0.0, scale=0.01, size=vector_size)
    
    def actualizaPesos(self, X, y):
        """
        'y' son 1 o -1, de modo que si se predice bien update = 0
        (1 - 1) o (-1 - -1)
        """
        error = 0
        for xi, target in zip(X, y):
            update = self.m * (target - self.predecir(xi))
            self.w[1:] += update * xi
            self.w[0] += update
            error += int(update != 0.0)
        self.errors.append(error)

    def net_input(self, X):
        """Calculate net input: XW + w[0] """
        return numpy.dot(X, self.w[1:]) + self.w[0]

    def predecir(self, X):
        """Devuelve el valor de la funcion de la neurona, la escalon, 1 o 0"""
        return numpy.where(self.net_input(X) >= 0.0, 1, -1)