import pandas
import numpy

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