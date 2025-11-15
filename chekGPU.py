# import tensorflow as tf
# def iiii():
#
#     print("TensorFlow version:", tf.__version__)
#     print("GPU доступны:", tf.config.list_physical_devices('GPU'))
# import subprocess
#
# def check_gpu():
#     try:
#         result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, shell=True)
#         if result.returncode == 0:
#             print("✅ Видеокарта NVIDIA обнаружена!")
#             lines = result.stdout.split('\n')
#             for i in range(min(8, len(lines))):
#                 print(lines[i])
#             return True
#         else:
#             print("❌ nvidia-smi не сработал")
#             return False
#     except Exception as e:
#         print(f"❌ Ошибка: {e}")
#         print("Вероятно, у вас нет видеокарты NVIDIA или драйверы не установлены")
#         return False
#
# check_gpu()
# iiii()
# import os
# import subprocess
#
# print("Проверка установки CUDA...")
#
# # Проверяем наличие CUDA 11.2
# cuda_path_11_2 = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin\nvcc.exe"
# if os.path.exists(cuda_path_11_2):
#     print("✅ CUDA 11.2 найдена")
# else:
#     print("❌ CUDA 11.2 не найдена по пути:", cuda_path_11_2)
#
# # Проверяем cuDNN
# cudnn_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\include\cudnn.h"
# if os.path.exists(cudnn_path):
#     print("✅ cuDNN найден")
# else:
#     print("❌ cuDNN не найден")
# import os
#
# print("Проверка необходимых DLL файлов...")
#
# cuda_bin = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin"
#
# # Критически важные DLL для TensorFlow 2.10
# required_dlls = [
#     "cudart64_110.dll",      # CUDA Runtime
#     "cudnn64_8.dll",         # cuDNN
#     "cublas64_11.dll",       # cuBLAS
#     "cublasLt64_11.dll",     # cuBLAS Lt
#     "cufft64_10.dll",        # cuFFT
#     "curand64_10.dll",       # cuRAND
#     "cusolver64_11.dll",     # cuSolver
# ]
#
# print(f"Проверка в папке: {cuda_bin}")
# for dll in required_dlls:
#     dll_path = os.path.join(cuda_bin, dll)
#     if os.path.exists(dll_path):
#         print(f"✅ {dll}")
#     else:
#         print(f"❌ {dll} - ОТСУТСТВУЕТ!")
#
# # Проверка PATH
# print(f"\nПроверка PATH...")
# path = os.environ.get('PATH', '')
# cuda_paths = [p for p in path.split(';') if 'CUDA' in p and 'v11.2' in p]
# if cuda_paths:
#     print("✅ Пути CUDA 11.2 в PATH:")
#     for p in cuda_paths:
#         print(f"  {p}")
# else:
#     print("❌ Пути CUDA 11.2 не найдены в PATH")
#
# import os
#
# # Принудительно добавляем пути CUDA 11.2 в PATH
# cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2"
# cuda_bin = os.path.join(cuda_path, "bin")
# cuda_lib = os.path.join(cuda_path, "lib", "x64")
#
# # Добавляем в начало PATH
# os.environ['PATH'] = f"{cuda_bin};{cuda_lib};" + os.environ['PATH']
#
# print("Добавлены пути CUDA в PATH:")
# print(f"BIN: {cuda_bin}")
# print(f"LIB: {cuda_lib}")
#
# # Теперь импортируем tensorflow
# import tensorflow as tf
#
# print(f"TensorFlow version: {tf.__version__}")
# print(f"GPU доступны: {tf.config.list_physical_devices('GPU')}")

# import os
# import sys
# import subprocess
# import ctypes
# from ctypes import wintypes
#
# print("=" * 70)
# print("ГЛУБОКАЯ ДИАГНОСТИКА TENSORFLOW + CUDA")
# print("=" * 70)
#
# # 1. Принудительная настройка путей ДО всего
# cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2"
# os.environ['PATH'] = f"{cuda_path}\\bin;{cuda_path}\\lib\\x64;" + os.environ['PATH']
# os.environ['CUDA_PATH'] = cuda_path
#
# # 2. Проверка загрузки DLL
# print("🔍 Проверка загрузки критических DLL...")
# dll_checks = [
#     ("cudart64_110.dll", "CUDA Runtime"),
#     ("cudnn64_8.dll", "cuDNN"),
#     ("cublas64_11.dll", "cuBLAS"),
# ]
#
# for dll_name, description in dll_checks:
#     try:
#         dll_path = os.path.join(cuda_path, "bin", dll_name)
#         ctypes.CDLL(dll_path)
#         print(f"✅ {description} ({dll_name}) - загружена")
#     except Exception as e:
#         print(f"❌ {description} ({dll_name}) - ошибка: {e}")
#
# # 3. Проверка через nvcc
# print("\n🔍 Проверка компилятора nvcc...")
# try:
#     result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, timeout=10)
#     if result.returncode == 0:
#         print("✅ nvcc доступен")
#         for line in result.stdout.split('\n'):
#             if "release" in line.lower():
#                 print(f"   Версия: {line.strip()}")
#     else:
#         print("❌ nvcc не работает")
# except Exception as e:
#     print(f"❌ nvcc не найден: {e}")
#
# # 4. Проверка переменных среды TensorFlow
# print("\n🔍 Переменные среды TensorFlow:")
# tf_vars = {k: v for k, v in os.environ.items() if 'TF_' in k or 'CUDA' in k.upper()}
# for k, v in tf_vars.items():
#     print(f"   {k}={v}")
#
# # 5. Импорт TensorFlow с детальной диагностикой
# print("\n🔍 Импорт TensorFlow...")
# try:
#     import tensorflow as tf
#
#     print(f"✅ TensorFlow {tf.__version__} импортирован")
#
#     # Детальная информация о сборке
#     print("\n📋 Информация о сборке TensorFlow:")
#     build_info = tf.sysconfig.get_build_info()
#     for key, value in build_info.items():
#         print(f"   {key}: {value}")
#
#     # Проверка всех физических устройств
#     print("\n🔍 Все физические устройства:")
#     devices = tf.config.list_physical_devices()
#     for device in devices:
#         print(f"   {device}")
#
#     # Специальная проверка GPU
#     print("\n🎯 Специальная проверка GPU:")
#     gpus = tf.config.list_physical_devices('GPU')
#     if gpus:
#         print(f"🎉 НАЙДЕНО GPU: {len(gpus)}")
#         for i, gpu in enumerate(gpus):
#             print(f"   GPU {i}: {gpu}")
#             try:
#                 details = tf.config.experimental.get_device_details(gpu)
#                 print(f"     Детали: {details}")
#             except:
#                 print(f"     Детали: недоступны")
#     else:
#         print("❌ GPU не найдены")
#
# except Exception as e:
#     print(f"❌ Ошибка импорта TensorFlow: {e}")
#     import traceback
#
#     traceback.print_exc()
#
# print("=" * 70)
#
# import tensorflow as tf
#
# print("=" * 50)
# print("ФИНАЛЬНАЯ ПРОВЕРКА GPU")
# print("=" * 50)
#
# build_info = tf.sysconfig.get_build_info()
# print("📋 Информация о сборке:")
# print(f"   is_cuda_build: {build_info.get('is_cuda_build', 'N/A')}")
# print(f"   cuda_version: {build_info.get('cuda_version', 'N/A')}")
# print(f"   cudnn_version: {build_info.get('cudnn_version', 'N/A')}")
#
# gpus = tf.config.list_physical_devices('GPU')
# print(f"\n🎯 GPU устройства: {gpus}")
#
# if gpus:
#     print("🎉 УСПЕХ! TensorFlow с GPU поддержкой работает!")
#     # Тест производительности
#     with tf.device('/GPU:0'):
#         import time
#         start = time.time()
#         a = tf.random.normal([1000, 1000])
#         b = tf.random.normal([1000, 1000])
#         c = tf.matmul(a, b)
#         print(f"✅ Тест GPU выполнен за: {time.time() - start:.3f} сек")
# else:
#     print("❌ Проблема: TensorFlow без GPU поддержки")
#
# print("=" * 50)

import tensorflow as tf
import sys
import os


def check_gpu_setup():
    """Проверка настройки GPU для проекта"""
    print("=" * 60)
    print("ПРОВЕРКА GPU ДЛЯ ПРОЕКТА")
    print("=" * 60)

    # 1. Информация о системе
    print("📋 Информация о системе:")
    print(f"Python: {sys.version}")
    print(f"TensorFlow: {tf.__version__}")

    # 2. Проверка физических устройств
    print("\n🔍 Физические устройства:")
    gpus = tf.config.list_physical_devices('GPU')
    cpus = tf.config.list_physical_devices('CPU')

    print(f"GPU: {len(gpus)} устройств")
    print(f"CPU: {len(cpus)} устройств")

    if gpus:
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu}")
            try:
                details = tf.config.experimental.get_device_details(gpu)
                print(f"    Детали: {details}")
            except:
                print(f"    Детали: недоступны")

    # 3. Проверка доступности GPU для TensorFlow
    print("\n🎯 Доступность GPU для TensorFlow:")
    if tf.test.is_gpu_available():
        print("✅ TensorFlow может использовать GPU")
        gpu_devices = tf.config.experimental.list_physical_devices('GPU')
        for device in gpu_devices:
            print(f"  Доступно: {device}")
    else:
        print("❌ TensorFlow не может использовать GPU")

    # 4. Тест производительности
    print("\n🧪 Тест производительности GPU:")
    if gpus:
        try:
            with tf.device('/GPU:0'):
                # Большая операция для теста GPU
                size = 5000
                a = tf.random.normal([size, size])
                b = tf.random.normal([size, size])

                import time
                start_time = time.time()
                c = tf.matmul(a, b)
                gpu_time = time.time() - start_time

                print(f"✅ Умножение матриц {size}x{size}: {gpu_time:.2f} сек")
                print(f"✅ Результат: {c.shape}")

        except Exception as e:
            print(f"❌ Ошибка теста GPU: {e}")
    else:
        print("❌ Тест GPU не выполнен - нет доступных GPU")

    # 5. Проверка памяти GPU
    print("\n💾 Память GPU:")
    if gpus:
        try:
            from tensorflow.python.client import device_lib
            devices = device_lib.list_local_devices()
            for device in devices:
                if device.device_type == 'GPU':
                    memory_info = device.memory_limit
                    print(f"  Доступно памяти: {memory_info / 1024 ** 3:.1f} GB")
        except:
            print("  Информация о памяти: недоступна")

    print("=" * 60)
    return len(gpus) > 0


if __name__ == "__main__":
    gpu_available = check_gpu_setup()
    if gpu_available:
        print("\n🎉 GPU ГОТОВА К РАБОТЕ В ПРОЕКТЕ!")
    else:
        print("\n⚠️  ВНИМАНИЕ: GPU не доступна, проект будет работать на CPU")