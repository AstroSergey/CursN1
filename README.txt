Пакет программ для работы калькулятора экспозиций + пайплайн начальной калибровки

calibrator.ipynb
Пайплайн начальной калибровки
не содержит рабочих модулей align и platesolve
----------------------------------------
ParB.csv, ParG.csv, ParR.csv 
файлы содержат данные для построения модели аппроксимации параметров SN, F, t в соответствующих столбцах где:
SN - сигнал/шум
F - инструментальный поток в электронах(считается через формулы связи с учетом воздушной массы)
t - время экспозиции
	P.S. лучше конечно заменить F на B, V, R, AM чтобы при уточнении формул перехода эти данные оставались опорными
Данные таблицы можно дополнять со свох наблюдений, однако лучше для каждого сета состовлять свои и разделить модели для некоторых значений seeing'a
--------------------------------------------
subprog.py
Программа в которой составлялись таблицы данных для построения моделей
Программа содержит 3 модели аппроксимации, в итоговый калькулятор переносится функция лучше описывающая t для заданных в таблицах параметрах
	P.S. авто-переноса нет, в конечных стоит модель нелинейного коэффициента
Результаты модели можно просматривать через датафрейм(print(df))
Есть подмодуль для просмотра точек модели
Есть подмодуль ручного тестирования t(F,SN)
-----------------------------------------------
modelB.txt, modelG.txt, modelR.txt
файл содержащий коффициенты текущей модели
является результатом программы subprog.py
требует запуска программы subprog.py для своего обновления(если база данных модели была обновлена)
------------------------------------------
MAINPROG.py
Программа калькулятора
--------------------------------------------
Остальное - игнорируем
