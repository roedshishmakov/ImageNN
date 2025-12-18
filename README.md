# **Neural Network Image Classifier**

Проект представляет собой реализацию нейронной сети для классификации изображений, написанную на Python. Система способна обучаться на пользовательских датасетах, сохранять и загружать модели, а также классифицировать новые изображения.

### Основные возможности:

-Полноценная реализация нейронной сети с нуля (без использования готовых фреймворков)

-Поддержка различных архитектур через конфигурационные файлы

-Множество функций активации: ReLU, Sigmoid, Softmax, Transparent

-Сохранение и загрузка моделей в формате HDF5

-Визуализация процесса обучения через графики потерь

-Классификация рукописных цифр из пользовательских изображений

-Дообучение существующих моделей

-Гибкая система командной строки

### Структура проекта:

- main.py              # Основные классы нейронной сети
- activations.py       # Функции активации
- ImageNN.py          # Основной интерфейс командной строки
- tools.py            # Вспомогательные функции
- exceptions.py       # Пользовательские исключения
- default_train.py    # Скрипт обучения на MNIST
- requirements.txt    # Заимствования

### Установка:

Предварительные требования:

- Python 3.7 или выше
- pip (менеджер пакетов Python)

Установка библиотек
``` bash
pip install -r requirements.txt
```

### Использование
**Основные команды:**
1. Обучение модели с кастомной конфигурацией (-y или --train):
``` bash
python3 ImageNN.py --train <model_name> <config_file> <train_dataset_path>
```
Пример:

``` bash
python3 ImageNN.py --train my_model config1 dataset/train
```
2. Простое обучение (-s или --simple-train):
``` bash
python3 ImageNN.py --simple-train <model_name> <epochs> <train_dataset_path>
```

3. Загрузка и тестирование модели (-l или --load):
``` bash
python3 ImageNN.py --load <model_name> <test_images_path>
```

При тестировании модель выводит:

- Имя тестового файла

- Предсказанный класс

4. Дообучение существующей модели (-f или --fine):
``` bash
python3 ImageNN.py --fine <model_name> <additional_epochs> <new_dataset_path>
```

5. Просмотр графика потерь (-sl или --show-loss):
``` bash
python3 ImageNN.py --show-loss <model_name>
```

6. Или с любой командой (-g или --graph):
``` bash
python3 ImageNN.py -l -g <model_name> <test_images_path>
```

7. Создание шаблона конфигурации
``` bash
python ImageNN.py --create-config config_name 
```

8. Помощь
``` bash
python ImageNN.py --help
```


## Формат файла конфигурации

``` ini
#Конфигурация нейронной сети
#Формат: [layer_type]:[size]:[activation]:[use_bias]:[random_radius]

#Скрытые и выходные слои:
dense:32:relu:true:0.1
dense:32:relu:true:0.1
dense:10:softmax:false:0.1

#Параметры обучения
learning_rate=0.1
epochs=10
clip_value=5.0
use_cross_entropy=true
```

## Формат датасета
**Формат:** jpg/png

**Размер изображений:** любой

**Имена файлов:** {class_label}_{unique_id}.{extension}

Пример:
``` text
dataset/
├── 0_image1.png
├── 0_image2.jpg
├── 1_image1.png
├── 1_image2.jpg
├── 2_image1.png
└── ...
```

## Пример создания и обучения пользовательской модели
Создайте конфигурационный файл:

``` bash
python ImageNN.py --create-config my_config
```
Отредактируйте конфигурацию в configs/my_config.config

``` ini
# Конфигурация нейронной сети
# Формат: [layer_type]:[size]:[activation]:[use_bias]:[random_radius]

dense:32:relu:true:0.1
dense:32:relu:true:0.1
dense:10:sigmoid:false:0.1

# Параметры обучения
learning_rate=0.1
epochs=10
clip_value=5.0
use_cross_entropy=False
```

Обучите модель:
``` bash
python ImageNN.py --train my_model my_config dataset/train
```

После обучения можно просмотреть график потерь:
``` bash
python ImageNN.py --graph model_name
```

Протестируйте модель:
``` bash
python ImageNN.py --load my_model dataset/test
```

При желании дообучим модель на еще одном датасете:
``` bash
python3 ImageNN.py --fine my_model 15 my_new_dataset
```

Посмотрим новый график потерь:
``` bash
python3 ImageNN.py -sl my_model
```

## Структура сохранения
Проект создает следующие директории:

``` text
ImageNN/
├── weight_saves/      # Сохраненные модели (.h5)
├── loss_saves/       # Файлы с потерями (.txt)
├── configs/          # Конфигурационные файлы (.config)
└── dataset/          # Пользовательские датасеты
```


## Проект включает систему пользовательских исключений:
- ArgumentError - ошибки в аргументах командной строки
- PathError - ошибки путей к файлам
- EpochError - ошибки в количестве эпох
- ValidationError - ошибки валидации
- IncorrectCommand - некорректные команды