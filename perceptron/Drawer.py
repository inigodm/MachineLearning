"""
Leemos los datos del csv de valores de las flores que que tenemos
"""
import numpy
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap  
    
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

