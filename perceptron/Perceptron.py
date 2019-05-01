import numpy
import pandas
import matplotlib.pyplot as plt

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

"""
Leemos los datos del csv de valores de las flores que que tenemos
"""
from matplotlib.colors import ListedColormap

class Data(object):
    """
    En esta clase obtenemos todos los datos ue tenemos de las florecicas esas
    """
    # Columnas de datos:
    L_SEPALO = 0
    W_SEPALO = 1
    L_PETALO = 2
    W_PETALO = 3
    NOMBRE_FLOR = 4

    def __init__(self):
        self.calcX_y()

    def load_csv(self, url):
        return pandas.read_csv(url)

    def get_sub_group(self, data, files_from, files_to, columns):
        return data.iloc[files_from:files_to, columns]

    def calcX_y(self):
        df = self.load_csv('./iris.data')
        self.y = self.get_sub_group(df, 0, 100, [self.NOMBRE_FLOR])
        self.y = numpy.where(self.y == 'Iris-setosa', -1, 1)
        self.X = numpy.array(self.get_sub_group(df, 0, 100, [self.L_SEPALO, self.L_PETALO]))
    
    
class Drawer(object):
    def draw_plots(self, data):
        """
        Dibujamos los dos tipos de flores por separado, sabemos que los
        primeros 50 datos corresponden a un tipo de flor y los siguientes
        50 a otra. En este caso los datos estan ordenados y por eso se cojen asi
        """
        plt.scatter(data.X[:50,0], data.X[:50,1], color='red', marker="o", label="setosa")
        plt.scatter(data.X[50:100, 0], data.X[50:100,1], color='blue', marker="x", label="versicolor")
        return self

    def plot_decision_regions(self, data, classifier, resolution=0.02):
        markers = ('s', 'x', 'o', '^', 'v')
        colors = ('red', 'blue', 'ligthgreen', 'gray', 'cyan')
        cmap = ListedColormap(colors[:len(numpy.unique(data.y))])
        # representar la superficie de decision
        x1_m, x1_M = data.X[:, 0].min() - 1, data.X[:, 0].max() + 1
        x2_m, x2_M = data.X[:, 1].min() - 1, data.X[:, 1].max() + 1
        xx1, xx2 = numpy.meshgrid(numpy.arange(x1_m, x1_M, resolution),
                                  numpy.arange(x2_m, x2_M, resolution))
        Z = classifier.predecir(numpy.array([xx1.ravel(), xx2.ravel()]).T)
        print(classifier.w)
        Z = Z.reshape(xx1.shape)
        plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(xx2.min(), xx2.max())
        return self

    def drawLine(self, a, b, xmin, xmax, resolution=0.02):
        resy = []
        resx = []
        for x in numpy.arange(xmin, xmax, resolution):
            resy.append(a*x+b)
            resx.append(x)
        plt.plot(resx, resy, linewidth=2.0, color="green")
        return self



    def draw(x_label="sepal length (cm)", y_label="petal length (cm)", loc='upper left'):
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend(loc=loc)
        plt.show()


data = Data()
"""
Ahora mostramos como evoluciona el numero de errores con las epocas en nuestro Perceptron
Vemos que a la quinta epoca dejamos de tener errores y parece que la neurona esta entrenada
"""
ppn = Perceptron(m = 0.1, n_iter = 10)
ppn.entrenar(data.X, data.y)
"""
Ahora hacemos una funcion para dibujar el limite de decision
"""
drawer = Drawer()
drawer.plot_decision_regions(data, classifier=ppn)
drawer.draw_plots(data)
# se superpone la recta de 0 = w2x2 + w1x1 + w0 => x2 = -w1x1/w2 - w0/w2 con xmin y xmax.
drawer.drawLine(-ppn.w[1]/ppn.w[2], -ppn.w[0]/ppn.w[2], data.X[:, 0].min() - 1, data.X[:, 0].max() + 1)
drawer.draw()