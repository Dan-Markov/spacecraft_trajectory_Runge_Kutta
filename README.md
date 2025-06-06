# Моделирование влияния верхней атмосферы на движение КА 🛰️

## 📌 Описание проекта
Численное моделирование возмущенного движения космического аппарата (КА) на низкой околоземной орбите с учетом аэродинамического торможения в верхних слоях атмосферы.

### Ключевые особенности:
- **Модель атмосферы**: Динамическая по ГОСТ Р 25645.166-2004 (учет индекса солнечной активности F107)
- **Метод интегрирования**: Рунге-Кутта 4-го порядка
- **Визуализация**: 3D-траектория и график высоты 
 

## 📊 Исходные данные
| Параметр               | Значение       |
|------------------------|----------------|
| Начальная высота       | 276 км         |
| Наклонение орбиты      | 75°            |
| Индекс F107            | 75             |
| Баллистический коэффициент | 0.004 м²/кг |

## 📂 Файлы
- `main.py` – Python-код с комментариями.  
- `requirments.txt` – файл для установки библиотек, требующихся для выполнения кода.  
- `Отчёт.pdf` – Полный отчёт по проведённой работе.  

## 🛠 Технологический стек
```python
Python 3.10
NumPy - векторные вычисления, линейная алгребра
Matplotlib - визуализация результатов
```

## 🚀 Запуск программы
Предварительные требования
Перед запуском необходимо установить зависимости:

### Установка библиотек (выполнить один из вариантов):
Через командную строку (CMD/Bash):
```bash
pip install numpy matplotlib
```
Или с использованием файла requirements.txt:
```bash
pip install -r requirements.txt
```
### Варианты запуска
- **Запуск через командную строку (CMD/Bash)**
```bash
python main.py
```
- **Запуск в PyCharm**
1. Откройте проект в PyCharm
2. Убедитесь, что интерпретатор Python настроен:

File → Settings → Project: <ваш проект> → Python Interpreter

3.Выберите существующее виртуальное окружение или создайте новое

- Установите зависимости через PyCharm:
- В терминале PyCharm выполните команды установки (см. выше)
- Или через интерфейс: нажмите на красный значок в правом верхнем углу → Install requirements

4.Запустите скрипт:

Правой кнопкой на main.py → Run
