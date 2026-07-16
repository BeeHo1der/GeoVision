# GeoVision AI: модель, checkpoints и работа на Windows

Эта инструкция рассчитана на обычную командную строку Windows — `cmd.exe`.
Команды PowerShell и Linux в неё не смешиваются.

## 1. Что находится в вашем best.pth

Checkpoint был разобран без выполнения содержащегося в нём Pickle-кода.

| Параметр | Значение |
|---|---:|
| Имя | `best.pth` |
| Размер | 293 545 163 байта, около 280 MiB |
| Формат | PyTorch ZIP checkpoint |
| Эпоха | 45 |
| Начальная эпоха после resume | 26 |
| Архитектура | U-Net с ResNet-34 encoder |
| Вход | 3 канала |
| Выход | 2 класса |
| Класс палеорусла | индекс 1 |
| Число тензоров модели | 278 |
| Loss | `Combined clDICE` |
| Validation loss | 0,115758 |
| IoU класса палеорусла | 0,895533 |
| Dice | 0,944790 |
| mIoU | 0,945781 |

Внутри находятся:

- `model_state_dict` — веса нейросети;
- `optimizer_state_dict` — состояние оптимизатора для продолжения обучения;
- `epoch`, `start_epoch` — эпохи;
- `val_loss`, `val_metrics` — метрики;
- `loss_type` — название loss;
- `original_task_id` — идентификатор эксперимента ClearML.

Для сайта нужен только `model_state_dict`. Оптимизатор занимает много места и
при инференсе не используется.

## 2. Почему появилась ошибка weights_only

Начиная с PyTorch 2.6, `torch.load()` по умолчанию использует:

```python
weights_only=True
```

Ваш файл содержит NumPy-значения в `val_metrics`, поэтому ограниченный
загрузчик останавливается на `numpy._core.multiarray.scalar`.

Есть два способа работы:

1. Разрешить полную Pickle-загрузку доверенного `best.pth`.
2. Один раз открыть доверенный файл и создать безопасный inference-checkpoint
   без оптимизатора и NumPy-объектов.

Для сайта рекомендуется второй способ.

Никогда не используйте `weights_only=False` для случайных checkpoint из
интернета: Pickle-файл теоретически может выполнить произвольный код.

## 3. Правильный запуск проекта в cmd.exe

Откройте «Командную строку». Перейдите в папку проекта:

```cmd
cd /d C:\Users\User\Desktop\geology_assistant_mvp
```

Проверьте содержимое:

```cmd
dir
```

В списке должны быть `main.py`, `requirements.txt`, папки `static` и
`templates`.

### Создание виртуального окружения

```cmd
python -m venv .venv
```

В обычной командной строке активировать окружение нужно так:

```cmd
call .venv\Scripts\activate.bat
```

Команда `.venv\Scripts\Activate.ps1` предназначена для PowerShell. По вашему
логу она не активировала окружение: `pip` устанавливал библиотеки в Miniconda.

После активации строка должна начинаться с `(.venv)`. Проверьте Python:

```cmd
where python
python -m pip --version
```

Первый путь должен вести в:

```text
C:\Users\User\Desktop\geology_assistant_mvp\.venv\Scripts\python.exe
```

### Установка зависимостей

```cmd
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Если не хотите активировать окружение, можно всегда обращаться к нему явно:

```cmd
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 4. Куда положить best.pth

Скопируйте файл в корень проекта:

```text
C:\Users\User\Desktop\geology_assistant_mvp\best.pth
```

Проверьте:

```cmd
dir best.pth
```

Размер должен быть примерно 293,5 МБ.

## 5. Создание файла .env в cmd.exe

В `cmd.exe` используется команда `copy`, а не `Copy-Item` и не `cp`:

```cmd
copy /Y .env.example .env
```

Откройте файл:

```cmd
notepad .env
```

Для первого запуска с оригинальным checkpoint содержимое должно быть таким:

```text
MODEL_PATH=best.pth
MODEL_INPUT_SIZE=512
MODEL_ENCODER=resnet34
MODEL_IN_CHANNELS=3
MODEL_CLASSES=2
POSITIVE_CLASS_INDEX=1
IMAGENET_NORMALIZATION=true
ALLOW_DEMO_MODE=true
TRUST_MODEL_CHECKPOINT=true
CPU_THREADS=4
WARMUP_MODEL=true
```

`TRUST_MODEL_CHECKPOINT=true` разрешается только потому, что это ваш собственный
checkpoint.

## 6. Проверка checkpoint

Остановите сайт сочетанием `Ctrl+C`, если он запущен. Затем выполните:

```cmd
python checkpoint_tool.py inspect best.pth --trust
```

Ожидаемые основные строки:

```text
Model tensors: 278
Input channels: 3
Output classes: 2
epoch: 45
loss_type: Combined clDICE
Contains optimizer state: True
```

## 7. Создание облегчённой модели для сайта

Выполните:

```cmd
python checkpoint_tool.py convert best.pth best_inference.pth --trust
```

Утилита:

- загрузит доверенный training checkpoint;
- извлечёт `model_state_dict`;
- перенесёт веса на CPU;
- удалит состояние оптимизатора;
- сохранит архитектуру и основные метрики;
- создаст `best_inference.pth`.

Исходный `best.pth` не удаляется и остаётся пригодным для продолжения обучения.

После конвертации измените `.env`:

```text
MODEL_PATH=best_inference.pth
TRUST_MODEL_CHECKPOINT=false
```

Остальные строки оставьте без изменений.

## 8. Запуск сайта

```cmd
python main.py
```

Или без активации окружения:

```cmd
.venv\Scripts\python.exe main.py
```

Откройте:

```text
http://127.0.0.1:8000
```

Для проверки модели откройте:

```text
http://127.0.0.1:8000/api/status
```

Успешное состояние содержит:

```json
{
  "mode": "model",
  "model_loaded": true,
  "architecture": "U-Net + resnet34",
  "output_classes": 2,
  "positive_class_index": 1
}
```

Если интерфейс показывает «Демонстрационный режим», наведите курсор на статус
или откройте `/api/status`: поле `error` содержит точную причину.

## 9. Как модель обрабатывает изображение

1. Изображение переводится в RGB.
2. Масштабируется до `512 × 512`.
3. Значения пикселей переводятся в диапазон 0–1.
4. Применяется ImageNet-нормализация:

```text
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

5. U-Net возвращает два канала logits.
6. Применяется `softmax` по каналам.
7. Канал с индексом `1` используется как вероятность палеорусла.
8. Вероятность масштабируется до исходного размера изображения.
9. U-Net сама назначает каждому пикселю бинарный класс и строит маску; ползунка
   порога в интерфейсе нет.

## 10. Как работать с checkpoints дальше

Для обучения и сайта нужны разные файлы.

### Training checkpoint

Нужен для продолжения обучения. Сохраняйте:

```python
torch.save(
    {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "val_loss": val_loss,
        "val_metrics": {key: float(value) for key, value in metrics.items()},
        "loss_type": "Combined clDICE",
    },
    "last_train_checkpoint.pth",
)
```

Этот файл используется только в обучающем скрипте.

### Best training checkpoint

Обновляйте его только при улучшении главной метрики:

```python
if val_dice > best_dice:
    best_dice = val_dice
    torch.save(checkpoint, "best_train_checkpoint.pth")
```

Можно использовать IoU вместо Dice, но главная метрика должна быть одна и та
же во всех экспериментах.

### Inference checkpoint

Для сайта сохраняйте только CPU-веса и простые метаданные:

```python
cpu_state = {
    key: value.detach().cpu()
    for key, value in model.state_dict().items()
}

torch.save(
    {
        "model_state_dict": cpu_state,
        "architecture": "segmentation_models_pytorch.Unet",
        "encoder_name": "resnet34",
        "classes": 2,
        "positive_class_index": 1,
        "input_size": 512,
        "normalization": "imagenet",
    },
    "best_inference.pth",
)
```

Именно `best_inference.pth` переносится на компьютер, сервер или в Docker с
сайтом.

## 11. Продолжение обучения из checkpoint

В обучающем скрипте:

```python
checkpoint = torch.load(
    "last_train_checkpoint.pth",
    map_location=device,
    weights_only=False,
)

model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

if checkpoint.get("scheduler_state_dict"):
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

start_epoch = checkpoint["epoch"] + 1
```

Используйте `weights_only=False` только для checkpoint, созданного вашей
командой.

## 12. Как дальше изменять сайт

Откройте проект в Visual Studio Code:

```cmd
code .
```

Основные файлы:

| Файл | Что изменять |
|---|---|
| `templates/index.html` | поля, кнопки и структура страницы |
| `static/styles.css` | цвета, размеры, карточки, адаптивность |
| `static/app.js` | загрузка файлов и отображение результатов |
| `main.py` | API и последовательность анализа |
| `model_service.py` | архитектура и инференс модели |
| `calculations.py` | объём запасов и стоимость |
| `well_planner.py` | алгоритм координат скважин |

После изменения Python-файла Uvicorn перезапустит сервер. После изменения CSS
или JavaScript обновите браузер сочетанием `Ctrl+F5`.

Перед изменениями создайте рабочую копию или Git-репозиторий:

```cmd
git init
git add .
git commit -m "Initial GeoVision MVP"
```

Большие модели не нужно добавлять в Git. Создайте `.gitignore` и добавьте:

```text
.venv/
__pycache__/
*.pyc
*.pth
.env
```

## 13. Быстрая диагностика

### Сайт не запускается

```cmd
where python
python -m pip --version
python -m pip install -r requirements.txt
```

### Модель не загрузилась

```cmd
python checkpoint_tool.py inspect best.pth --trust
```

Затем проверьте `/api/status`.

### Изменения интерфейса не появились

Нажмите `Ctrl+F5` или откройте сайт в приватном окне.

### Порт занят

```cmd
python -m uvicorn main:app --host 127.0.0.1 --port 8001 --reload
```

Откройте `http://127.0.0.1:8001`.

### Остановка сайта

В окне командной строки нажмите `Ctrl+C`.

### После загрузки изображения бесконечно крутится индикатор

Используйте актуальные `main.py`, `well_planner.py` и `static/app.js`. В новой
версии расчёт скважин автоматически переводится на оптимизированную сетку, а в
браузер передаются уменьшенные предпросмотры вместо четырёх полноразмерных PNG.

После обновления файлов перезапустите сервер и нажмите `Ctrl+F5`. В карточке
результата будет показано общее время обработки. Подробное время этапов также
возвращается полем `timings_seconds` в ответе API.

При запуске актуальной версии дождитесь строк:

```text
Warming up model on cpu with 512x512 input...
Model warmup completed.
Application startup complete.
```

Для отдельного измерения скорости выполните при остановленном сайте:

```cmd
python diagnose_model.py
```

Или передайте путь к реальному полихрому:

```cmd
python diagnose_model.py "C:\путь\к\image.png"
```

Утилита напечатает время загрузки, прогрева и одного инференса.
