import numpy as np
from datetime import datetime
from astropy.coordinates import EarthLocation, SkyCoord, AltAz
from astropy.time import Time
from astropy.io import fits
from astropy import units as u
from math import cos
from math import radians
from scipy.optimize import root_scalar
import tkinter as tk
from tkinter import ttk

# Раздел функций
def tograd(t):                                          # функция переводящая ra[h, m, s] в градусы
    return t[0]*15 + t[1]/60*15 + t[2]/3600*15


def RGB(x, y, z, am):                                   # функция расчета уравнения перехода от BVR к RGB с учетом экстицнии
    A = np.array([                                      # x = Bio, y = Gio, z = Rio
        [1.165, -0.165, 0],
        [0, 0.88, 0.12],
        [0, -0.378, 1.378]
    ])
    B = np.array([x-21.0612, y-20.7355, z-20.1309])     # x - Cb, y - Cv, z - Cr
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


def filt(f):
    if f == 'B':
        return 0
    elif f == 'G':
        return 1
    elif f == 'R':
        return 2

def AM_count(ra_ha, dec_deg):
    ra_deg = tograd(ra_ha) * u.deg

    obs_location = EarthLocation.from_geodetic(lat=latitude, lon=longitude, height=obs_height)       # location of obervations

    # observation time
    obs_time = Time(datetime(time_obs[0], time_obs[1], time_obs[2], time_obs[3], time_obs[4], time_obs[5]),scale='utc',location=obs_location)   # возможна ошибка на 3часа

    star_ICRS = SkyCoord(ra=ra_deg, dec=dec_deg)
    star_altaz = star_ICRS.transform_to(AltAz(location=obs_location, obstime=obs_time))
    z = 90 - star_altaz.alt.deg
    AM = 1/cos(radians(z))
    return(AM)

def model_with_nonlinear_correction(params, S, F):                                                                          # функция модели
    k, C0, C1, C2, a, b = params
    C = C0 + C1 * F**a + C2 * S**b
    t_approx = ((S - C) / (k * np.sqrt(F)))**2
    return t_approx



# config files
# magnitudes в B, V, R (Джонсоновские)
B, V, R = 12.26, 11.48, 11.48                                   # звездные величины в B, V(Джонсоновские), R
latitude, longitude, obs_height = 43.6781, 41.4597, 2070          # координаты наблюдателя       
time_obs = [2024, 4, 8, 21, 12, 10]                             # Время наблюдений [год, месяц, день, час, минута, секунда]
ra_ha = [12, 42, 53]                                            # значения ra в часах, arcmin, arcsec
dec_deg = -0.495555 * u.deg                                     # значение dec в градусах
SN = 10                                                         # Signal / Noise
Filter = "B"

file = open('model' + Filter + '.txt', 'r')
coeff = file.readlines()
coeff = coeff[0].split(',')
for i in range(len(coeff)):
    coeff[i] = float(coeff[i])
file.close()

def submit():
    try:
        # Получаем данные из полей ввода
        global B, V, R, time_obs, ra_ha, dec_deg, SN, Filter

        B = float(entry_B.get())
        V = float(entry_V.get())
        R = float(entry_R.get())

        # Разбираем дату
        date_str = entry_date.get() # Формат: "год-месяц-день"
        year, month, day = map(int, date_str.split('-'))

        # Разбираем время
        time_str = entry_time.get() # Формат: "часы:минуты:секунды"
        hour, minute, second = map(int, time_str.split(':'))
        time_obs = [year, month, day, hour, minute, second]

        # Разбираем RA
        ra_str = entry_ra.get() # Формат: "часы минуты секунды"
        ra_hours, ra_minutes, ra_seconds = map(float, ra_str.split())
        ra_ha = [ra_hours, ra_minutes, ra_seconds]

        # Получаем DEC
        dec_deg = float(entry_dec_deg.get()) * u.deg # Преобразуем значение DEC в формат с u.deg

        # Получаем S/N и фильтр
        SN = int(entry_SN.get())
        Filter = combo_filter.get()

        # Выполняем расчет (пример, можно заменить на ваши формулы)
        F = Flux(RGB(B, V, R, AM_count(ra_ha, dec_deg))[filt(Filter)])
        result = model_with_nonlinear_correction(coeff, SN, F)

        # Вывод результата в графическом интерфейсе
        result_label.config(text=f"Результат расчетов: {result}")
    except Exception as e:
        result_label.config(text=f"Ошибка: {e}", fg="red")

# Создаем окно
root = tk.Tk()
root.title("Параметры наблюдений")

# Создаем поля ввода
labels = [
    "B", "V", "R", "Дата (год-месяц-день)", "Время (часы:минуты:секунды)",
    "RA (часы минуты секунды)", "DEC (градусы)",
    "Signal / Noise (SN)", "Фильтр",
    ]

entries = {}
for i, label_text in enumerate(labels):
    label = ttk.Label(root, text=label_text)
    label.grid(row=i, column=0, sticky="w", padx=5, pady=2)
    
    if label_text == "Фильтр":
        combo_filter = ttk.Combobox(root, values=["B", "V", "R"], state="readonly")
        combo_filter.grid(row=i, column=1, padx=5, pady=2)
        combo_filter.set("B")
    else:
        entry = ttk.Entry(root)
        entry.grid(row=i, column=1, padx=5, pady=2)
        entries[label_text] = entry

# Сопоставляем значения полей с переменными
entry_B = entries["B"]
entry_V = entries["V"]
entry_R = entries["R"]
entry_date = entries["Дата (год-месяц-день)"]
entry_time = entries["Время (часы:минуты:секунды)"]
entry_ra = entries["RA (часы минуты секунды)"]
entry_dec_deg = entries["DEC (градусы)"]
entry_SN = entries["Signal / Noise (SN)"]

# Кнопка "Подтвердить"
submit_button = ttk.Button(root, text="Подтвердить", command=submit)
submit_button.grid(row=len(labels), column=0, columnspan=2, pady=10)

# Метка для результата
result_label = ttk.Label(root, text="")
result_label.grid(row=len(labels) + 1, column=0, columnspan=2)

# Запуск приложения
root.mainloop()





