from Data import Data
from Drawer import Drawer
from Neuronas import Perceptron
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