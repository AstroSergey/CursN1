import numpy as np
from datetime import datetime
from astropy.coordinates import EarthLocation, SkyCoord, AltAz
from astropy.time import Time
from astropy.io import fits
from astropy import units as u

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from scipy.optimize import root_scalar
from scipy.optimize import curve_fit
from scipy.optimize import minimize

import pandas as pd


# Раздел функций
def tograd(t):                                         # функция переводящая ra[h, m, s] в градусы
    return t[0]*15 + t[1]/60*15 + t[2]/3600*15

def RGB(x, y, z, am):                                  # функция расчета уравнения перехода от BVR к RGB с учетом экстицнии
    A = np.array([                                     # x = Bio, y = Gio, z = Rio
        [1.165, -0.165, 0],
        [0, 0.88, 0.12],
        [0, -0.378, 1.378]
    ])
    B = np.array([x-21.0612, y-20.7355, z-20.1309]) # x - Cb, y - Cv, z - Cr
    sol = np.linalg.solve(A, B)
    x, y, z = sol
    Bi = x + (0.2758 - 0.0037 * (x - y)) * am
    Gi = y + (0.2049 + 0.0054 * (y - z)) * am
    Ri = z + (0.1518 - 0.011 * (y - z)) * am
    return Bi, Gi, Ri

def Flux(m):
    """
    Вычисляет x из формулы m = -2.5 * log10(x).

    Параметры:
    m (float или массив): Значение m.

    Возвращает:
    x (float или массив): Значение x.
    """
    return 10**(-m / 2.5)

    # Пример использования
    # m_values = [-1, 0, 1, 2.5]  # Пример значений m
    # x_values = compute_x(np.array(m_values))
    # print("m:", m_values)
    # print("x:", x_values)

def toSN(SNR):
    return 10 ** (SNR/10)

def S_from_SN(SN):

    # Решает уравнение G = x / sqrt(x + 100) для заданного значения G.

    # Parameters:
    # G (float): Значение G.

    # Returns:
    # float: Найденное значение x.

    # Проверка на допустимость G

    # Уравнение, которое нужно решить: G - x / sqrt(x + 100) = 0
    def equation(x):
        return SN - x / (x + 100)**0.5

    # Используем численное решение. Начальное приближение — 1.
    solution = root_scalar(equation, bracket=[0, 1e6], method='brentq')

    if solution.converged:
        return solution.root
    else:
        raise RuntimeError("Решение не найдено.")

    # Пример использования:
    # G = 700
    # x = solve_equation(G)
    # print(f"Решение: x = {x}")

def filt(f):
    if f == 'B':
        return 0
    elif f == 'G':
        return 1
    elif f == 'R':
        return 2

'''
# модуль который из старых таблиц B.csv, G.csv R.csv получал текущие таблицы
Filt = 'R'
file = open(Filt + '.csv', 'r')                           # файл с необходимыми данными (отдельно для B, G, R)
lines = file.readlines()
file.close



for i in range(len(lines)):                         # a = i + 1, b = 4j + 2 + k
    lines[i] = lines[i].split(',')

T = np.zeros((len(lines)-1, 10, 4))                           # Time x STARnumber x Param(B, V, R, S/N)
for i in range(len(T)):                             # по строкам +1 (по моментам наблюдений)
    for j in range(10):                             # по столбцам с разделением +2(по звездам)
        for k in range(4):                          # по параметрам звезд
            T[i][j][k] = float(lines[i+1][4*j + 2 + k])

Tt = np.zeros((len(lines)-1, 10, 3))
for i in range(len(Tt)):
    for j in range(len(Tt[0])):
        Tt[i][j][0] = (RGB(T[i][j][0], T[i][j][1], T[i][j][2], float(lines[i+1][1]))[filt(Filt)])    # инструментальная звездная величина[0]=B, [1]=G, [2]=R
        Tt[i][j][1] = toSN(T[i][j][3])                                                      # Отношение сигнал/шум
        Tt[i][j][2] = 90                                                                    # время экспозиции



t, SN, F = [], [], []
for i in range(len(Tt)):
    for j in range(len(Tt[0])):
        F.append(Flux(Tt[i][j][0]))
        SN.append(Tt[i][j][1])
        t.append(Tt[i][j][2])


# модуль который запишет нужные штуки в нужном формата, но что-то голова не соображает уже, не к спеху
# должен получиться файл из 3-х строчек в 1 - F, 2 - SN, 3 - t
file_path = 'Par' + Filt + '.csv'
with open(file_path, mode='w', encoding='utf-8') as file:
    for values in zip(F, SN, t):
        file.write(','.join(map(str, values)) + '\n')
    file.close
'''


'''
# модуль построения точек на графике (m * SNR * expose)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = Tt[:, :, 0]
y = Tt[:, :, 1]
z = Tt[:, :, 2]

ax.scatter(x, y, z, color='black')
plt.show()
'''
Filt = 'B'

file = open('Par' + Filt + '.csv', 'r')             # F, SN, t
lines = file.readlines()
file.close


F, SN, t = [], [], []
for i in range(len(lines)):
    lines[i] = lines[i].split(',')
    F.append(float(lines[i][0]))
    SN.append(float(lines[i][1]))
    t.append(float(lines[i][2]))
data = {                            # Заполнение датафрейма
    "SN": SN,                       # Значения S
    "t": t,                         # Значения t
    "F": F                          # Значения F
}

df = pd.DataFrame(data)             # Создание датафрейма

# --------------------------------------------
# Общая модель: t = (SN / (k * sqrt(F)))**2
# --------------------------------------------
# Модель с поправочным членом
# SN = k * sqrt(t) * sqrt(F) + C


def model(sqrt_t, k, C):                                                # функция модели с поправкой
    return k * sqrt_t + C


df["sqrt_t"] = np.sqrt(df["t"])                                         # sqrt(t) для известных t


popt, _ = curve_fit(model, np.sqrt(df["F"]) * df["sqrt_t"], df["SN"])   # Аппроксимация k и C


k, C = popt                                                             # Необходимые параметры модели


df["t_approx_corrected"] = ((df["SN"] - C) / (k * np.sqrt(df["F"])))**2 # Расчет t по модели

# --------------------------------------------
# Модель с параметрической поправкой(масштабирование С)
# С = С0 + С1*F + C2*SN



def model_with_correction(params, S, F):                                                                # функция модели с динамической C
    k, C0, C1, C2 = params
    C = C0 + C1 * F + C2 * S
    t_approx = ((S - C) / (k * np.sqrt(F)))**2
    return t_approx


def loss_function(params):                                                                              # функция минимизации параметра
    t_approx = model_with_correction(params, df["SN"], df["F"])
    return np.sum((df["t"] - t_approx)**2)

                                                                                                        # Оптимизация
initial_guess = [1.0, 0.0, 0.0, 0.0]                                                                    # Начальные значения для k, C0, C1, C2
result = minimize(loss_function, initial_guess)


k_opt, C0_opt, C1_opt, C2_opt = result.x                                                                
df["t_approx_dynamic_C"] = model_with_correction([k_opt, C0_opt, C1_opt, C2_opt], df["SN"], df["F"])    # Вычисление t в модели

# ---------------------------------------------
# Модель с нелинейной поправкой
# C = C0 + C1*F**a + C2*S**b




def model_with_nonlinear_correction(params, S, F):                                                                          # функция модели
    k, C0, C1, C2, a, b = params
    C = C0 + C1 * F**a + C2 * S**b
    t_approx = ((S - C) / (k * np.sqrt(F)))**2
    return t_approx


def loss_function_nonlinear(params):                                                                                        # Целевая функция
    t_approx = model_with_nonlinear_correction(params, df["SN"], df["F"])
    return np.sum((df["t"] - t_approx)**2)

                                                                                                                            # Оптимизация
initial_guess = [1.0, 0.0, 0.0, 0.0, 0.5, 0.5]                                                                              # Начальные значения для k, C0, C1, C2, a, b
result_nonlinear = minimize(loss_function_nonlinear, initial_guess)

k_nl, C0_nl, C1_nl, C2_nl, a_nl, b_nl = result_nonlinear.x                                                                  # Необходимые параметры модели

model_par = open('model' + Filt + '.txt', 'w')
model_par.write(str(k_nl) + ',' + str(C0_nl) + ',' + str(C1_nl) + ',' + str(C2_nl) + ',' + str(a_nl) + ',' + str(b_nl))





df["t_approx_nonlinear_C"] = model_with_nonlinear_correction([k_nl, C0_nl, C1_nl, C2_nl, a_nl, b_nl], df["SN"], df["F"])    # Уточненные значения t

print(df)



# подмодуль проверки функции
SN_input = 50
F_input = 450
t_Calc = model_with_nonlinear_correction([k_nl, C0_nl, C1_nl, C2_nl, a_nl, b_nl], SN_input, F_input)
print(t_Calc)


'''
# подмодуль просмотра расхождения расчитанных моделью
# Реальные t и аппроксимированные t
plt.scatter(df["t"], df["t_approx_dynamic_C"], label="Динамическая С", color="blue")
plt.scatter(df["t"], df["t_approx_corrected"], label="Постоянная С", color="green")
plt.scatter(df["t"], df["t_approx_nonlinear_C"], label="Нелинейная С", color="orange")
plt.plot(df["t"], df["t"], label="Идеальная зависимость", color="red", linestyle="--")
plt.xlabel("Реальное t")
plt.ylabel("Аппроксимированное t")
plt.legend()
plt.show()
'''







print('debug')