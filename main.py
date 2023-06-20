import cv2
import numpy as np
import math
import multiprocessing
import time

def load_izo(fname,n): #
    """Функция загрузки изображения и разбиения его на области для параллельной обработки"""
    if fname.endswith('jpeg') or fname.endswith('jpg') or fname.endswith('bmp') or fname.endswith('gif') or fname.endswith('tiff') or  fname.endswith('pnm') or fname.endswith('png'): # Загрузка локального изображения
        img = cv2.imread(fname)
        rows = img.shape[0]
        columns = img.shape[1]

        # Коэффициент сжатия изо в зависимости от размера
        if (rows+columns <= 650*2):
            coef_sj = 1
        elif (rows+columns > 650*2) and (rows+columns <= 1050*2):
            coef_sj= 1.5
        elif (rows+columns > 1050*2) and (rows+columns <= 2050*2):
            coef_sj = 2.5
        elif (rows+columns > 2050*2) and (rows+columns <= 3450*2):
            coef_sj = 4.0
        elif (rows+columns > 3450*2) and (rows+columns <= 5050*2):
            coef_sj = 5.5
        elif (rows+columns > 5050*2) and (rows+columns <= 7850*2):
            coef_sj = 8.5
        else: coef_sj = 10.5

        # Сжатие изо
        coef_sj = coef_sj*1.2
        img = cv2.resize(img, dsize=(int(columns/coef_sj), int(rows/coef_sj)), interpolation=cv2.INTER_CUBIC)
        # Размерность массива сжатого изображения (количество строк и столбцов исходной матрицы изображения)
        rows    = img.shape[0]
        columns = img.shape[1]
        # Разбиение изображения на n областей для параллельной обработки
        # Матрица 2x2 начала и конца области изо для каждой из n областей изображения по строкам(элементы матрицы 11(начало),12(конец))
        # и столбцам (элементы матрицы 21(начало),22(конец))
        a = np.empty([2, n * 2], dtype=np.int_)
        if not ((n != 1) and (n!=2) and (n!=3) and (n!=4) and (n!=6) and (n!=9) and (n!=16)):
            flag_proc = 1
            for j in range(n*2):
                if (n >= 1) and (n <= 3):
                    if (rows >= columns):
                        a[0, j] = math.ceil(j / 2) * (rows / n)
                        a[1, j] = int((j + 1) % 2 == 0) * columns
                    else:
                        a[0, j] = int((j + 1) % 2 == 0) * rows
                        a[1, j] = math.ceil(j / 2) * (columns / n)
                elif (n == 4):
                    # 4 квадрата 2на2
                    if (j < math.sqrt(n)*2):
                        a[0, j] = math.ceil(j / 2) * (rows / math.sqrt(n))
                        a[1, j] = int((j + 1) % 2 == 0) * (columns / math.sqrt(n))
                    else:
                        a[0, j] = math.ceil((j-math.sqrt(n)*2) / 2) * (rows / math.sqrt(n))
                        a[1, j] = (int((j + 1) % 2 == 0) + 1) * (columns / math.sqrt(n))
                elif (n == 6):
                    if (rows >= columns):
                        if (j < n):
                            a[0, j] = math.ceil(j / 2) * (rows / 3)
                            a[1, j] = int((j + 1) % 2 == 0) * (columns / 2)
                        else:
                            a[0, j] = math.ceil((j-n) / 2) * (rows / 3)
                            a[1, j] = (int((j + 1) % 2 == 0) + 1) * (columns / 2)
                    else:
                        if (j < n):
                            a[0, j] = int((j + 1) % 2 == 0) * (rows / 2)
                            a[1, j] = math.ceil(j / 2) * (columns / 3)
                        else:
                            a[0, j] = (int((j + 1) % 2 == 0) + 1) * (rows / 2)
                            a[1, j] = math.ceil((j-n) / 2) * (columns / 3)
                elif (n == 9):
                    # 9 квадратов 3на3
                    if (j < math.sqrt(n) * 2):
                        a[0, j] = math.ceil(j / 2) * (rows / math.sqrt(n))
                        a[1, j] = int((j + 1) % 2 == 0) * (columns / math.sqrt(n))
                    elif (j >= math.sqrt(n) * 2) and (j < 2 * math.sqrt(n) * 2):
                        a[0, j] = math.ceil((j - math.sqrt(n) * 2) / 2) * (rows / math.sqrt(n))
                        a[1, j] = (int((j + 1) % 2 == 0) + 1) * (columns / math.sqrt(n))
                    else:
                        a[0, j] = math.ceil((j - 2 * math.sqrt(n) * 2) / 2) * (rows / math.sqrt(n))
                        a[1, j] = (int((j + 1) % 2 == 0) + 2) * (columns / math.sqrt(n))
                elif (n == 16):
                    # 16 квадратов 4на4
                    if (j < math.sqrt(n) * 2):
                        a[0, j] = math.ceil(j / 2) * (rows / math.sqrt(n))
                        a[1, j] = int((j + 1) % 2 == 0) * (columns / math.sqrt(n))
                    elif (j >= math.sqrt(n) * 2) and (j < 2 * math.sqrt(n) * 2):
                        a[0, j] = math.ceil((j - math.sqrt(n) * 2) / 2) * (rows / math.sqrt(n))
                        a[1, j] = (int((j + 1) % 2 == 0) + 1) * (columns / math.sqrt(n))
                    elif (j >= 2 * math.sqrt(n) * 2) and (j < 3 * math.sqrt(n) * 2):
                        a[0, j] = math.ceil((j - 2 * math.sqrt(n) * 2) / 2) * (rows / math.sqrt(n))
                        a[1, j] = (int((j + 1) % 2 == 0) + 2) * (columns / math.sqrt(n))
                    else:
                        a[0, j] = math.ceil((j - 3 * math.sqrt(n) * 2) / 2) * (rows / math.sqrt(n))
                        a[1, j] = (int((j + 1) % 2 == 0) + 3) * (columns / math.sqrt(n))
        else:
            flag_proc = 0
    return rows, columns, img, a, flag_proc

class img_proc:
    """Класс сортировки пикселей изображения по палитре цветов и яркости"""
    def __init__(self, img):
        self.img = img
    def img_pix_color(self):
        rows = self.img.shape[0]
        columns = self.img.shape[1]

        # Массивы для чисел типа int
        color_type = np.empty([rows, columns], dtype=int)
        # Массивы для чисел типа float
        color_phase = np.empty([rows, columns], dtype=float)
        colors_diap1 = np.empty([11, 2], dtype=float)

        # Расчет фазы цвета на цветовом круге для каждого пикселя изображения
        for i in range(rows):
            for j in range(columns):

                R = int(self.img[i, j, 0])
                G = int(self.img[i, j, 1])
                B = int(self.img[i, j, 2])

                Xc = G * math.cos(30 * deg_2_rad)
                Yc = R - (G * math.cos(60 * deg_2_rad))
                Xd = Xc - (B * math.cos(30 * deg_2_rad))
                Yd = Yc - (B * math.cos(60 * deg_2_rad))

                color_phase[i, j] = math.atan2(Xd, Yd) * rad_2_deg
                if color_phase[i, j] < 0: color_phase[i, j] = 360 + color_phase[i, j]

        # Сортировка пикселей по цвету
        # Круговая палитра цветов (фаза от 0 до 360 градусов)
        colors_1 = np.array([0, 30, 60, 90, 150, 180, 210, 240, 270, 320, 360])

        for i in range(len(colors_1)):
            colors_diap1[i, 0] = colors_1[i - 1]
            colors_diap1[i, 1] = colors_1[i]

        colors_diap1 = colors_diap1[1:11]

        for i in range(rows):
            for j in range(columns):
                d2 = list((color_phase[i, j] >= colors_diap1[:, 0]) * (color_phase[i, j] < colors_diap1[:, 1]))
                try:
                    b = d2.index(True)
                except:
                    b = 0
                color_type[i, j] = b

        return color_type

    def img_pix_bright(self):
        rows = self.img.shape[0]
        columns = self.img.shape[1]
        # Количество яркостей для сортировки
        bright_sort = 8

        # Массивы для чисел типа int
        bright_type = np.empty([rows, columns], dtype=int)
        # Массивы для чисел типа float
        bright_otn = np.empty([rows, columns], dtype=float)
        brights_diap = np.empty([bright_sort, 2], dtype=float)

        # Расчет яркости
        for i in range(rows):
            for j in range(columns):
                b = int(self.img[i, j, 0]) + int(self.img[i, j, 1]) + int(self.img[i, j, 2])
                bright_otn[i, j] = b * 100 / (255 * 3)

        # Сортировка пикселей по яркости
        brights = np.arange(0, 100, 100 / bright_sort)
        for i in range(len(brights)):
            brights_diap[i, 0] = brights[i - 1]
            brights_diap[i, 1] = brights[i]
        brights_diap = np.roll(brights_diap, [-1, -1])

        for i in range(rows):
            for j in range(columns):
                d1 = list((bright_otn[i, j] >= brights_diap[:, 0]) * (bright_otn[i, j] < brights_diap[:, 1]))
                try:
                    a = d1.index(True)
                except:
                    a = bright_sort
                bright_type[i, j] = a
        return bright_type

async def img_pix_color_bright(res):
    return await func(res)

def visualization(res, bright_type, color_type):# Функция визуализации

    cv2.namedWindow('Image')
    cv2.imshow('Image',res)

    rows    = res.shape[0]
    columns = res.shape[1]

    res2 = np.array(res)
    for i in range(rows):
        for j in range(columns):
            if bright_type[i, j] in bright_need and color_type[i, j] in color_need:
                res2[i, j, :] = res[i, j, :]
            else:
                res2[i, j, :] = 255
    cv2.namedWindow('Image2')
    cv2.imshow('Image2', res2)
    cv2.waitKey(0)

class Mymultiproc(multiprocessing.Process):
    """Класс параллельной обработки n областей изображения"""
    def __init__(self, func, img, zone11, zone12, zone21, zone22, rows, columns, an_array_X, X_np, an_array_Y, Y_np, name=''):
        multiprocessing.Process.__init__(self)
        self.name, self.func, self.img  = name, func, img
        self.zone11, self.zone12, self.zone21, self.zone22 = zone11, zone12, zone21, zone22
        self.rows, self.columns  = rows, columns
        self.an_array_X, self.X_np, self.an_array_Y,self.Y_np = an_array_X, X_np, an_array_Y, Y_np

    def run(self):
        # Объявление класса img_proc, присвоение его переменной self.res
        self.res = self.func(self.img[self.zone11:self.zone12,self.zone21:self.zone22,:])

        # Вызов функции цветовой сортировки пикселей из класса img_proc
        self.colors  = self.res.img_pix_color()
        # Запись результата работа функции в shared memory для multiprocessing
        self.X_np = np.frombuffer(self.an_array_X.get_obj(), dtype=np.float64).reshape(self.rows, self.columns)
        self.X_np[self.zone11:self.zone12, self.zone21:self.zone22] = self.colors

        # Вызов функции сортировки пикселей по яркости из класса img_proc
        self.brights = self.res.img_pix_bright()
        # Запись результата работа функции в shared memory для multiprocessing
        self.Y_np = np.frombuffer(self.an_array_Y.get_obj(), dtype=np.float64).reshape(self.rows, self.columns)
        self.Y_np[self.zone11:self.zone12, self.zone21:self.zone22] = self.brights

if __name__ == '__main__':
    start_time = time.time()
    ## Константы
    deg_2_rad = math.pi / 180
    rad_2_deg = 180 / math.pi

    # Словарь цветов пикселей
    dict_color = {'синий': 0,'сине-голубой': 1,'голубой': 2,'зеленый': 3,'желто-зеленый': 4,'желтый': 5,'оранжевый': 6,
            'красный': 7,'розовый': 8,'фиолетовый': 9}

    ## Параметры, вводимые пользователем
    # Имя файла
    fname = '2012.jpeg'
    # Количество используемых ядер процессора
    n = 4
    # Список необходимых цветов пикселей на изображении
    color_need  = [5,6] # 0...9
    # Список необходимых яркостей пикселей на изображении
    bright_need = [0,1,2,3,4,5,6,7,8] # 0...8

    [rows, columns, img, zones, flag_proc] = load_izo(fname, n)
    if flag_proc == 0:
        print('Значение n должно быть равно 1,2,3,4,6,9 или 16')
    else:
        # Сортировка пикселей по яркости и цветности - пустые массивы
        color_type_all = np.empty([rows, columns], dtype=int)
        bright_type_all = np.empty([rows, columns], dtype=int)
        # инфа по расшариванию памяти массива в мультипоточных приложениях здесь:
        # https: // stackoverflow.com / questions / 65199943 / python - multiprocessing - when - share - a - numpy - array
        X_shape = (rows, columns)
        data = np.empty([rows, columns], dtype=int)
        an_array_X = multiprocessing.Array('d', X_shape[0] * X_shape[1], lock=True)
        an_array_Y = multiprocessing.Array('d', X_shape[0] * X_shape[1], lock=True)
        # Wrap X as an numpy array so we can easily manipulates its data.
        X_np = np.frombuffer(an_array_X.get_obj()).reshape(X_shape)
        Y_np = np.frombuffer(an_array_Y.get_obj()).reshape(X_shape)
        # Copy data to our shared array.
        np.copyto(X_np, data)
        np.copyto(Y_np, data)

        # Создание n параллельных процессов в цикле, в каждом из которых обрабатывается свой участок изображения
        i = 1
        while i <= n:
            locals()['p' + str(i)] = Mymultiproc(img_proc, img, zones[0, (2 * i - 2)],zones[0, (2 * i - 1)], zones[1, (2 * i - 2)], zones[1, (2 * i - 1)],rows, columns, an_array_X, X_np, an_array_Y, Y_np,img_pix_color_bright.__name__)
            i += 1

        # Запуск n параллельных процессов в цикле, в каждом из которых обрабатывается свой участок изображения
        i = 1
        while i <= n:
            locals()['p' + str(i)].start()
            i += 1
        i = 1
        while i <= n:
            locals()['p' + str(i)].join()
            i += 1

        color_type_all = np.frombuffer(an_array_X.get_obj()).reshape(rows, columns)
        bright_type_all = np.frombuffer(an_array_Y.get_obj()).reshape(rows, columns)
        end_time = time.time() - start_time
        print(f'Время обработки = {round(end_time,2)} с')
        visualization(img, bright_type_all, color_type_all)