import os
import json

# Конфигурация для первого компьютера (главный worker)
TF_CONFIG_MASTER = {
    'cluster': {
        'worker': ['192.168.1.100:12300', '192.168.1.101:12301']
    },
    'task': {'type': 'worker', 'index': 0}
}

# Конфигурация для второго компьютера
TF_CONFIG_WORKER = {
    'cluster': {
        'worker': ['192.168.1.100:12300', '192.168.1.101:12301']
    },
    'task': {'type': 'worker', 'index': 1}
}

# На первом компьютере выполнить:
# os.environ['TF_CONFIG'] = json.dumps(TF_CONFIG_MASTER)

# На втором компьютере выполнить:
os.environ['TF_CONFIG'] = json.dumps(TF_CONFIG_WORKER)