import numpy as np
import math
from matplotlib import pyplot as plt

# Общие константы
MU_EARTH = 398600.45  # км³/с²
R_EARTH = 6371  # км

# Параметры варианта 7
h_initial = 276  # км
i_initial = np.radians(75)  # в радианах
F107 = 75
ballistic_coeff = 0.004  # м²/кг

# Датасет из ГОСТ Р 25645.166-2004 (сгенерирован с помощью ИИ, чтоб не вводить вручную). С помощью него можно найти плотность для различных F107
DATASET_GOST_ATMOSPHERE = {
    75: {
        'a0': 26.8629,
        'a1': -0.451674,
        'a2': 0.00290397,
        'a3': -1.06953e-5,
        'a4': 2.21598e-8,
        'a5': -2.42941e-11,
        'a6': 1.09926e-14
    },
    100: {
        'a0': 27.4598,
        'a1': -0.463668,
        'a2': 0.002974,
        'a3': -1.0753e-5,
        'a4': 2.17059e-8,
        'a5': -2.30249e-11,
        'a6': 1.00123e-14
    },
    125: {
        'a0': 28.6395,
        'a1': -0.490987,
        'a2': 0.00320649,
        'a3': -1.1681e-5,
        'a4': 2.36847e-8,
        'a5': -2.51809e-11,
        'a6': 1.09536e-14
    },
    150: {
        'a0': 29.6418,
        'a1': -0.514957,
        'a2': 0.00341926,
        'a3': -1.25785e-5,
        'a4': 2.5727e-8,
        'a5': -2.75874e-11,
        'a6': 1.21091e-14
    },
    175: {
        'a0': 30.1671,
        'a1': -0.527837,
        'a2': 0.00353211,
        'a3': -1.30227e-5,
        'a4': 2.66455e-8,
        'a5': -2.85432e-11,
        'a6': 1.25009e-14
    },
    200: {
        'a0': 29.7578,
        'a1': -0.517915,
        'a2': 0.00342699,
        'a3': -1.24137e-5,
        'a4': 2.48209e-8,
        'a5': -2.58413e-11,
        'a6': 1.09383e-14
    },
    250: {
        'a0': 30.7854,
        'a1': -0.545695,
        'a2': 0.00370328,
        'a3': -1.37072e-5,
        'a4': 2.80614e-8,
        'a5': -3.00184e-11,
        'a6': 1.31142e-14
    }
}


# Функция расчёта плотности атмосферы (ГОСТ Р 25645.166-2004)
# Аргументами являются только высота и индекс солнечной активности
def get_density(height_km, F107):
    K0 = 1 # т.к. F107 = F81. Остальные K не учитываем, т.к. они выражают влияние неучитываемых факторов
    rho_0 = 1.58868e-8
    a0 = DATASET_GOST_ATMOSPHERE[F107]['a0']
    a1 = DATASET_GOST_ATMOSPHERE[F107]['a1']
    a2 = DATASET_GOST_ATMOSPHERE[F107]['a2']
    a3 = DATASET_GOST_ATMOSPHERE[F107]['a3']
    a4 = DATASET_GOST_ATMOSPHERE[F107]['a4']
    a5 = DATASET_GOST_ATMOSPHERE[F107]['a5']
    a6 = DATASET_GOST_ATMOSPHERE[F107]['a6']
    rho_n = rho_0 * math.exp(a0 + a1 * height_km + a2 * height_km ** 2 + a3 * height_km ** 3 + a4 * height_km ** 4 + a5 * height_km ** 5 + a6 * height_km ** 6)
    density = rho_n * K0
    return density


# Функция вычисления производных состояния
def derivatives(statevec : np.array) -> np.array:
    derivs = np.zeros(shape=6)
    derivs[0:3] = statevec[3:6]
    r_cur = np.linalg.norm(statevec[0:3]) # Текущее расстояние от центра Земли в км
    h_cur = r_cur - R_EARTH # Текущая высота орбиты в км
    v_cur = np.linalg.norm(statevec[3:6])
    derivs[3:6] = - MU_EARTH / r_cur ** 3 * statevec[0:3] # Гравитационное ускорение (км/с^2)
    rho = get_density(h_cur, F107)  # кг/м³
    derivs[3:6] -= rho * ballistic_coeff * v_cur * statevec[3:6] * 1000 # Ускорение силы сопротивления набегающего потока (км/с^2)
    return derivs


# Метод Рунге-Кутты 4-го порядка
def runge_kutta4(statevec_i1 : np.array, step):
    statevec_i2 = np.zeros(shape=6)
    k1 = derivatives(statevec_i1)
    k2 = derivatives(statevec_i1 + step * k1 / 2)
    k3 = derivatives(statevec_i1 + step * k2 / 2)
    k4 = derivatives(statevec_i1 + step * k3)
    statevec_i2 += statevec_i1 + step * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return statevec_i2


def print_telemetry(time, statevec, height):
    print(
        f"t: {time:.0f} с; "
        f"X: {statevec[0]:.3f} км; "
        f"Y: {statevec[1]:.3f} км; "
        f"Z: {statevec[2]:.3f} км; "
        f"Vx: {statevec[3]:.1f} км/с; "
        f"Vy: {statevec[4]:.1f} км/с; "
        f"Vz: {statevec[5]:.1f} км/с; "
        f"Высота: {height:.3f} км"
    )


def solve(initial_statevec):
    current_state = initial_statevec.copy()
    time = 0.0
    revs = 0
    while True:
        h_current = np.linalg.norm(current_state[:3]) - R_EARTH
        if h_current <= target_height:
            break

        times.append(time)
        trajectory.append(current_state)
        heights.append(h_current)
        next_state = runge_kutta4(current_state, dt)
        if round(time, 2) % 21600 == 0:
            print_telemetry(time, current_state, h_current)

        if (current_state[2] <= 0 and next_state[2] >= 0):
            revs += 1

        time += dt
        current_state = next_state
    times.append(time)
    trajectory.append(current_state)
    heights.append(h_current)
    print_telemetry(time, current_state, h_current)
    print('РЕШЕНИЕ ЗАВЕРШЕНО')
    print(f'Число витков: {revs}')
    (print(f"Время снижения на 10 км: {times[-1] / 3600:.2f} часов"))


# Начальные условия для круговой орбиты
r_initial = R_EARTH + h_initial
v_initial = np.sqrt(MU_EARTH / r_initial)  # км/с

# Начальное состояние [x, y, z, vx, vy, vz] (км, км/с)
initial_state = np.array([
    r_initial, 0, 0,
    0, v_initial * np.cos(i_initial), v_initial * np.sin(i_initial)
    ], dtype=np.float64)

dt = 1.0  # шаг интегрирования

# Основное интегрирование
times = []
trajectory = []
heights = []
target_height = h_initial - 10  # 266 км


solve(initial_state)


# **ВЫВОД ДАННЫХ**
# График высоты
heights_plot = plt.figure().add_subplot()
plt.ticklabel_format(style='plain')
plt.xlabel("t, с")
plt.ylabel("h, км")
plt.minorticks_on()
plt.xlim([0., 371.])
plt.ylim([260., 280.])
plt.grid(which = 'major')
plt.grid(which = 'minor', linestyle = ':')
heights_plot.plot([elem / 3600 for elem in times[:]], heights[:], color = 'c')

# 3D траектория
trajectory = np.array(trajectory, dtype=np.float64)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2])
ax.set_xlabel('X (км)')
ax.set_ylabel('Y (км)')
ax.set_zlabel('Z (км)')
plt.title('Траектория КА')
ax.grid()
plt.show()